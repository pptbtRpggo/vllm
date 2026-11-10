# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rename a file inside a new encrypted 7z or RAR; keep the source.

Requires Python 3.9+ and 7-Zip or WinRAR. No third-party Python packages.
Example:
    encrypt_archive.py original.zip --name renamed.zip --output protected.7z
    encrypt_archive.py original.zip --backend winrar --output protected.rar
"""

import argparse
import getpass
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path


def validate_name(name):
    """Reject paths and names that cannot be ordinary Windows filenames."""
    invalid = '<>:"/\\|?*'
    if (
        not name
        or name in (".", "..")
        or name.endswith((" ", "."))
        or any(char in invalid or ord(char) < 32 for char in name)
    ):
        raise ValueError("新名字必须是合法文件名，不能包含路径或特殊字符。")
    stem = name.split(".", 1)[0].rstrip(" .").upper()
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(
        prefix + digit for prefix in ("COM", "LPT") for digit in "123456789¹²³"
    )
    if stem in reserved:
        raise ValueError("新名字不能使用 Windows 保留设备名。")
    return name


def find_7zip(explicit=None):
    """Find the console executable in PATH or normal Windows install paths."""
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"找不到 7-Zip 程序：{candidate}")
        return str(candidate)
    for name in ("7z.exe", "7zz.exe", "7z", "7zz", "7za"):
        found = shutil.which(name)
        if found:
            return found
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if root:
            for relative in ("7-Zip/7z.exe", "Programs/7-Zip/7z.exe"):
                candidate = Path(root) / relative
                if candidate.is_file():
                    return str(candidate)
    raise FileNotFoundError(
        "未找到 7-Zip。请先安装 https://www.7-zip.org/ ，"
        '或添加参数 --seven-zip "C:\\Program Files\\7-Zip\\7z.exe"。'
    )


def find_winrar(explicit=None):
    """Find WinRAR's console Rar.exe, not its GUI executable."""
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.name.lower() == "winrar.exe":
            candidate = candidate.with_name("Rar.exe")
        if candidate.name.lower() in ("unrar", "unrar.exe"):
            raise ValueError("UnRAR 只能解压，请指定 WinRAR 目录里的 Rar.exe。")
        if not candidate.is_file():
            raise FileNotFoundError(f"找不到 WinRAR 命令行工具 Rar.exe：{candidate}")
        return str(candidate)
    for name in ("Rar.exe", "rar.exe", "rar"):
        found = shutil.which(name)
        if found:
            return found
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if root:
            for relative in ("WinRAR/Rar.exe", "Programs/WinRAR/Rar.exe"):
                candidate = Path(root) / relative
                if candidate.is_file():
                    return str(candidate)
    raise FileNotFoundError(
        "未找到 WinRAR 的 Rar.exe。请使用 --winrar 指定路径，例如 "
        '--winrar "C:\\Program Files\\WinRAR\\Rar.exe"。'
    )


def select_backend(backend="auto", output=None, seven_zip=None, winrar=None):
    """Select the format explicitly, by extension, or by installed tools."""
    if seven_zip:
        if backend == "winrar" or winrar:
            raise ValueError("7-Zip 与 WinRAR 的选项不能同时指定。")
        backend = "7zip"
    if winrar:
        if backend == "7zip":
            raise ValueError("7-Zip 与 WinRAR 的选项不能同时指定。")
        backend = "winrar"
    if backend == "auto" and output:
        suffix = Path(output).suffix.lower()
        if suffix not in (".7z", ".rar"):
            raise ValueError("输出文件名必须以 .7z 或 .rar 结尾。")
        backend = "winrar" if suffix == ".rar" else "7zip"
    if backend == "winrar":
        return backend, find_winrar(winrar)
    if backend == "7zip":
        return backend, find_7zip(seven_zip)
    if backend != "auto":
        raise ValueError(f"未知的压缩工具：{backend}")
    try:
        return "7zip", find_7zip()
    except FileNotFoundError:
        return "winrar", find_winrar()


def run_7zip(executable, arguments, password, cwd=None):
    """Supply the password through stdin, never in the command line."""
    # Creation needs -p to enable encryption and ask for its password.
    # With 't'/'x', bare -p instead means an empty password: omit it to prompt.
    password_switch = ["-p"] if arguments[0] == "a" else []
    result = subprocess.run(
        [str(executable), *arguments, *password_switch, "-sccUTF-8"],
        input=((password + "\n") * 2).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        timeout=3600,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"7-Zip 操作失败（代码 {result.returncode}）：\n{detail}")


def encode_rar_password(password, encoding=None):
    """Match Rar.exe's ANSI decoding of redirected stdin on Windows."""
    if not password or any(char in password for char in "\r\n\x00"):
        raise ValueError("密码不能为空，也不能包含换行或 NUL 字符。")
    if len(password.encode("utf-16-le")) // 2 > 127:
        raise ValueError("WinRAR 密码最长 127 个字符，请使用较短的密码。")
    encoding = encoding or ("mbcs" if os.name == "nt" else "utf-8")
    try:
        encoded = password.encode(encoding, errors="strict")
        if encoded.decode(encoding, errors="strict") != password:
            raise ValueError("密码无法无损转换为当前系统编码。")
    except UnicodeError as error:
        raise ValueError(
            "密码含有当前 Windows 系统编码不支持的字符，请改用字母、数字或符号。"
        ) from error
    return encoded + b"\n"


def run_winrar(executable, arguments, password, cwd=None):
    """Use the console RAR password prompt without exposing it in argv."""
    result = subprocess.run(
        [str(executable), *arguments, "-scfr"],
        input=encode_rar_password(password),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        timeout=3600,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"WinRAR 操作失败（代码 {result.returncode}）：\n{detail}")


def publish(temporary, output):
    """Publish without overwriting a file created by another process."""
    if os.name == "nt":
        os.rename(temporary, output)  # Windows refuses an existing destination.
    else:
        os.link(temporary, output)  # Also refuses replacement on macOS/Linux.
        Path(temporary).unlink()


def encrypt_archive(source, output, inner_name, password, executable, backend="7zip"):
    """Wrap the renamed source in an encrypted archive and test it."""
    validate_name(inner_name)
    if not password or any(char in password for char in "\r\n\x00"):
        raise ValueError("密码不能为空，也不能包含换行或 NUL 字符。")
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if not source.is_file():
        raise ValueError("输入必须是一个文件。")
    if backend not in ("7zip", "winrar"):
        raise ValueError(f"未知的压缩工具：{backend}")
    extension = ".rar" if backend == "winrar" else ".7z"
    if output.suffix.lower() != extension:
        raise ValueError(f"输出文件名必须以 {extension} 结尾。")
    if backend == "winrar":
        encode_rar_password(password)
    if os.path.lexists(output):
        raise FileExistsError(f"输出已经存在，未覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the renamed unencrypted copy in the local temporary directory.
    # Only encrypted temporary output is written beside the destination.
    with (
        tempfile.TemporaryDirectory(prefix="archive_input_") as input_dir,
        tempfile.TemporaryDirectory(
            prefix="archive_output_", dir=output.parent
        ) as output_dir,
    ):
        renamed = Path(input_dir) / inner_name
        shutil.copyfile(source, renamed)
        temporary = Path(output_dir) / ("encrypted" + extension)
        member = "." + os.sep + inner_name
        if backend == "winrar":
            run_winrar(
                executable,
                [
                    "a",
                    "-cfg-",
                    "-ma5",
                    "-hp",
                    "-ep1",
                    "-m1",
                    "-y",
                    str(temporary),
                    member,
                ],
                password,
                cwd=input_dir,
            )
            run_winrar(executable, ["t", "-cfg-", "-p", str(temporary)], password)
        else:
            run_7zip(
                executable,
                ["a", "-t7z", "-mhe=on", "-mx=1", "-y", str(temporary), member],
                password,
                cwd=input_dir,
            )
            run_7zip(executable, ["t", str(temporary)], password)
        publish(temporary, output)
    return output


def main():
    """Run the Windows terminal interface."""
    parser = argparse.ArgumentParser(
        description="将已有文件改名后，套入加密 .7z 或 .rar；原文件保留。"
    )
    parser.add_argument("source", nargs="?", help="原压缩包路径")
    parser.add_argument("--name", help="外层包内的新文件名，包含原后缀")
    parser.add_argument("--output", help="新的加密 .7z 或 .rar 文件路径")
    parser.add_argument(
        "--backend",
        choices=("auto", "7zip", "winrar"),
        default="auto",
        help="压缩工具，默认自动查找",
    )
    programs = parser.add_mutually_exclusive_group()
    programs.add_argument("--seven-zip", help="手动指定 7z.exe 路径")
    programs.add_argument("--winrar", help="手动指定 WinRAR 目录中的 Rar.exe 路径")
    args = parser.parse_args()
    source = Path(args.source or input("原压缩包路径：").strip().strip('"'))
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    backend, executable = select_backend(
        args.backend, args.output, args.seven_zip, args.winrar
    )
    suffix = next(
        (
            s
            for s in (".tar.gz", ".tar.bz2", ".tar.xz")
            if source.name.lower().endswith(s)
        ),
        source.suffix,
    )
    suggested = secrets.token_hex(6) + suffix
    inner_name = args.name
    if inner_name is None:
        inner_name = input(f"包内新名字（保留后缀，回车用 {suggested}）：").strip()
        inner_name = inner_name or suggested
    validate_name(inner_name)
    extension = ".rar" if backend == "winrar" else ".7z"
    default_output = source.resolve().parent / (
        Path(inner_name).stem + "_加密" + extension
    )
    output = Path(args.output) if args.output else default_output
    print(f"压缩工具：{backend}\n包内文件名：{inner_name}\n输出文件：{output}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        password = getpass.getpass("新密码（输入时不显示）：")
        confirmation = getpass.getpass("再次输入新密码：")
    if password != confirmation:
        raise ValueError("两次密码不一致，未生成压缩包。")
    print("正在加密并检查完整性，请稍候……", flush=True)
    result = encrypt_archive(source, output, inner_name, password, executable, backend)
    print(f"完成：{result}\n原文件保留，未改名或删除。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已取消，原文件保留。", file=sys.stderr)
        sys.exit(130)
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
        getpass.GetPassWarning,
        EOFError,
    ) as error:
        print(f"失败：{error}", file=sys.stderr)
        sys.exit(1)
