# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encrypt a project, rename its archive, and compress it again.

Requires Python 3.9+ and 7-Zip or WinRAR. No third-party Python packages.
Directories: encrypted first archive -> random new name -> ordinary outer archive.
Use --encrypt-outer to encrypt the outer archive with the same password as well.
Files: preserve the original single-layer encrypted wrapping behavior.
Example:
    encrypt_archive.py D:\\project --backend winrar --output D:\\backup.rar
    encrypt_archive.py original.zip --name renamed.zip --output protected.7z
    encrypt_archive.py original.zip --backend winrar --output protected.rar
"""

import argparse
import getpass
import os
import secrets
import shutil
import stat
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


def run_7zip(executable, arguments, password=None, cwd=None):
    """Supply the password through stdin, never in the command line."""
    # Creation needs -p to enable encryption and ask for its password.
    # With 't'/'x', bare -p instead means an empty password: omit it to prompt.
    password_switch = ["-p"] if arguments[0] == "a" and password is not None else []
    result = subprocess.run(
        [str(executable), *arguments, *password_switch, "-sccUTF-8"],
        input=((password + "\n") * 2).encode("utf-8") if password is not None else b"",
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


def run_winrar(executable, arguments, password=None, cwd=None):
    """Use the console RAR password prompt without exposing it in argv."""
    result = subprocess.run(
        [str(executable), *arguments, "-scfr"],
        input=encode_rar_password(password) if password is not None else b"",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        timeout=3600,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"WinRAR 操作失败（代码 {result.returncode}）：\n{detail}")


def create_archive(source, output, password, executable, backend):
    """Archive one explicit file or directory without following symbolic links."""
    source, output = Path(source), Path(output)
    member = "." + os.sep + source.name
    if backend == "winrar":
        arguments = ["a", "-cfg-", "-ma5", "-ep1", "-m1", "-y"]
        if source.is_dir():
            arguments.extend(["-r", "-ol"])
        if password is not None:
            arguments.append("-hp")
        arguments.extend([str(output), member])
        run_winrar(executable, arguments, password, cwd=source.parent)
    else:
        arguments = ["a", "-t7z", "-mx=1", "-y", "-sse"]
        if source.is_dir():
            arguments.append("-snl")
        if password is not None:
            arguments.append("-mhe=on")
        arguments.extend([str(output), member])
        run_7zip(executable, arguments, password, cwd=source.parent)


def test_archive(archive, password, executable, backend):
    """Check an entire archive before publishing or using it as the next input."""
    if backend == "winrar":
        arguments = ["t", "-cfg-"]
        if password is not None:
            arguments.append("-p")
        arguments.append(str(archive))
        run_winrar(executable, arguments, password)
    else:
        run_7zip(executable, ["t", str(archive)], password)


def check_project_links(source):
    """Reject links whose restoration is not guaranteed by the 7z backend."""
    pending = [Path(source)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                reparse = getattr(info, "st_file_attributes", 0) & 0x400
                if stat.S_ISLNK(info.st_mode) or reparse:
                    raise ValueError(
                        f"7-Zip 目录模式不支持此链接或重解析点：{entry.path}。"
                        "请改用 --backend winrar，避免链接恢复不完整。"
                    )
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))


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
        create_archive(renamed, temporary, password, executable, backend)
        test_archive(temporary, password, executable, backend)
        publish(temporary, output)
    return output


def encrypt_project(
    source,
    output,
    inner_name,
    password,
    executable,
    backend="7zip",
    encrypt_outer=False,
):
    """Encrypt a complete directory, rename that archive, then compress it again."""
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if not source.is_dir() or source == source.parent:
        raise ValueError("输入必须是具体的项目目录，不能是磁盘根目录。")
    if backend not in ("7zip", "winrar"):
        raise ValueError(f"未知的压缩工具：{backend}")
    extension = ".rar" if backend == "winrar" else ".7z"
    validate_name(inner_name)
    if not inner_name.lower().endswith(extension) or output.suffix.lower() != extension:
        raise ValueError(f"第一层新名字和输出文件名都必须以 {extension} 结尾。")
    if not password or any(char in password for char in "\r\n\x00"):
        raise ValueError("密码不能为空，也不能包含换行或 NUL 字符。")
    if backend == "winrar":
        encode_rar_password(password)
    resolved_output = output.resolve()
    if resolved_output == source or source in resolved_output.parents:
        raise ValueError("输出必须放在项目目录外，避免把生成中的压缩包打包进自己。")
    if os.path.lexists(output):
        raise FileExistsError(f"输出已经存在，未覆盖：{output}")
    if backend == "7zip":
        check_project_links(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Both stages stay outside the source, even when TEMP points into the project.
    with tempfile.TemporaryDirectory(
        prefix="project_archive_", dir=output.parent
    ) as work_dir:
        work = Path(work_dir)
        first_dir, second_dir = work / "first", work / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first = first_dir / (source.name + extension)
        print("1/3：将整个项目加密压缩……", flush=True)
        create_archive(source, first, password, executable, backend)
        test_archive(first, password, executable, backend)
        renamed = work / inner_name
        first.rename(renamed)
        print(f"2/3：第一层压缩包已改名为 {inner_name}", flush=True)
        temporary = second_dir / ("outer" + extension)
        outer_password = password if encrypt_outer else None
        print("3/3：进行第二次压缩并检查完整性……", flush=True)
        create_archive(renamed, temporary, outer_password, executable, backend)
        test_archive(temporary, outer_password, executable, backend)
        publish(temporary, output)
    return output


def main():
    """Run the Windows terminal interface."""
    parser = argparse.ArgumentParser(
        description="项目目录先加密压缩，再随机改名并二次压缩；原项目保留。"
    )
    parser.add_argument("source", nargs="?", help="项目目录，或要封装的已有文件")
    parser.add_argument("--name", help="第一层压缩包的新名字，目录模式默认自动随机命名")
    parser.add_argument(
        "--output", help="最终外层 .7z 或 .rar 文件路径，放在项目目录外"
    )
    parser.add_argument(
        "--encrypt-outer",
        action="store_true",
        help="目录模式的第二层也使用同一密码加密",
    )
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
    source = Path(args.source or input("项目目录或文件路径：").strip().strip('"'))
    if not source.is_file() and not source.is_dir():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    backend, executable = select_backend(
        args.backend, args.output, args.seven_zip, args.winrar
    )
    extension = ".rar" if backend == "winrar" else ".7z"
    suffix = (
        extension
        if source.is_dir()
        else next(
            (
                s
                for s in (".tar.gz", ".tar.bz2", ".tar.xz")
                if source.name.lower().endswith(s)
            ),
            source.suffix,
        )
    )
    suggested = secrets.token_hex(6) + suffix
    inner_name = args.name
    if inner_name is None and source.is_dir():
        inner_name = suggested
    elif inner_name is None:
        inner_name = input(f"包内新名字（保留后缀，回车用 {suggested}）：").strip()
        inner_name = inner_name or suggested
    validate_name(inner_name)
    label = "_双层" if source.is_dir() else "_加密"
    default_output = source.resolve().parent / (
        Path(inner_name).stem + label + extension
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
    if source.is_dir():
        result = encrypt_project(
            source,
            output,
            inner_name,
            password,
            executable,
            backend,
            args.encrypt_outer,
        )
    else:
        result = encrypt_archive(
            source, output, inner_name, password, executable, backend
        )
    print(f"完成：{result}\n原项目/文件保留，未改名或删除。")
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
