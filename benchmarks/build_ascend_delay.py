# SPDX-License-Identifier: Apache-2.0
"""Build the TorchNPU submission adapter against the installed Torch ABI."""

import os
import sys
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load


def main():
    output = Path(sys.argv[1]).resolve()
    cann = Path(sys.argv[2]).resolve()
    npu = Path(torch_npu.__file__).resolve().parent
    source = Path(__file__).resolve().parents[1] / "csrc/ascend/hetero_delay_ops.cpp"
    os.environ.setdefault("MAX_JOBS", "1")
    library = load(
        name="hetero_delay",
        sources=[str(source)],
        extra_include_paths=[
            str(npu / "include"),
            str(npu / "include/third_party/acl/inc"),
            str(cann / "include"),
        ],
        extra_ldflags=[
            f"-L{npu / 'lib'}",
            "-ltorch_npu",
            f"-L{output}",
            "-lhetero_delay_kernels",
            f"-Wl,-rpath,{npu / 'lib'}",
            f"-Wl,-rpath,{output}",
        ],
        build_directory=str(output),
        is_python_module=False,
        verbose=True,
    )
    destination = output / "libhetero_delay.so"
    destination.unlink(missing_ok=True)
    destination.symlink_to(Path(library).name)
    print(destination)


if __name__ == "__main__":
    main()
