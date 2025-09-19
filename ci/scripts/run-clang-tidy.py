#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

def find_cu_files(root: Path):
    """Recursively yield all .cu files under root."""
    for path in root.rglob("*.cu"):
        if path.is_file():
            yield path

def run_clang_tidy_single(file: Path, clang_tidy: str, cuda_path: str, cuda_arch: str, includes: list[str]):
    """Run clang-tidy on a single file with optional include dirs."""
    cmd = [
        clang_tidy,
        "-warnings-as-errors='*'",
        "-header-filter='.*'",
        "-extra-arg=-Wno-unknown-cuda-version",
        "-extra-arg=-Wno-invalid-percent-escape",
        str(file),
        "--",
        "-x",
        "cuda",
        "-std=c++17",
        "--cuda-path=" + cuda_path,
        "--cuda-gpu-arch=sm_" + cuda_arch,
        "-D__CUDACC__",
        f"-I{cuda_path}/include",
    ]
    for inc in includes:
        cmd.extend([f"-I{inc}"])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(
        description="Recursively run clang-tidy on CUDA .cu files."
    )
    parser.add_argument("directory", help="Root directory to search.")
    parser.add_argument(
        "--clang-tidy", default="clang-tidy",
        help="Path to clang-tidy (default: clang-tidy from PATH)."
    )
    parser.add_argument(
        "--cuda-path", default="/usr/local/cuda",
        help="Path to CUDA installation (default: /usr/local/cuda)."
    )
    parser.add_argument(
        "--cuda-arch", default="80",
        help="CUDA architecture to use (default: 80)."
    )
    parser.add_argument(
        "-I", "--include", action="append", default=[],
        help="Include directory to pass to clang-tidy (can repeat)."
    )
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.exists() or not root.is_dir():
        print(f"error: '{root}' is not a directory", file=sys.stderr)
        sys.exit(2)

    cu_files = list(find_cu_files(root))
    if not cu_files:
        print("No .cu files found.")
        return

    failures = []
    for f in cu_files:
        ret = run_clang_tidy_single(f, args.clang_tidy, args.cuda_path, args.cuda_arch, args.include)
        if ret != 0:
            failures.append(f)

    print("\n=== Summary ===")
    print(f"Total: {len(cu_files)}  Succeeded: {len(cu_files)-len(failures)}  Failed: {len(failures)}")
    if failures:
        print("Failed files:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)

if __name__ == "__main__":
    main()
