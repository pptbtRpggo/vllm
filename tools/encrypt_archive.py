# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rename an archive inside a new password-protected 7z; keep the source.

Requires Python 3.9+ and an installed 7-Zip. No third-party Python packages.
Example:
    encrypt_archive.py original.zip --name renamed.zip --output protected.7z
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


def publish(temporary, output):
    """Publish without overwriting a file created by another process."""
    if os.name == "nt":
        os.rename(temporary, output)  # Windows refuses an existing destination.
    else:
        os.link(temporary, output)  # Also refuses replacement on macOS/Linux.
        Path(temporary).unlink()


def encrypt_archive(source, output, inner_name, password, executable):
    """Wrap the renamed source in an encrypted archive and test it."""
    validate_name(inner_name)
    if not password or any(char in password for char in "\r\n\x00"):
        raise ValueError("密码不能为空，也不能包含换行或 NUL 字符。")
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if not source.is_file():
        raise ValueError("输入必须是一个文件。")
    if output.suffix.lower() != ".7z":
        raise ValueError("输出文件名必须以 .7z 结尾。")
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
        temporary = Path(output_dir) / "encrypted.7z"
        run_7zip(
            executable,
            [
                "a",
                "-t7z",
                "-mhe=on",
                "-mx=1",
                "-y",
                str(temporary),
                "." + os.sep + inner_name,
            ],
            password,
            cwd=input_dir,
        )
        run_7zip(executable, ["t", str(temporary)], password)
        publish(temporary, output)
    return output


def main():
    """Run the Windows terminal interface."""
    parser = argparse.ArgumentParser(
        description="将已有压缩包改名后，套入新的加密 .7z；原文件保留。"
    )
    parser.add_argument("source", nargs="?", help="原压缩包路径")
    parser.add_argument("--name", help="外层包内的新文件名，包含原后缀")
    parser.add_argument("--output", help="新的加密 .7z 文件路径")
    parser.add_argument("--seven-zip", help="手动指定 7z.exe 路径")
    args = parser.parse_args()
    source = Path(args.source or input("原压缩包路径：").strip().strip('"'))
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    executable = find_7zip(args.seven_zip)
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
    default_output = source.resolve().parent / (Path(inner_name).stem + "_加密.7z")
    output = Path(args.output) if args.output else default_output
    print(f"包内文件名：{inner_name}\n输出文件：{output}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        password = getpass.getpass("新密码（输入时不显示）：")
        confirmation = getpass.getpass("再次输入新密码：")
    if password != confirmation:
        raise ValueError("两次密码不一致，未生成压缩包。")
    print("正在加密并检查完整性，请稍候……", flush=True)
    result = encrypt_archive(source, output, inner_name, password, executable)
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
