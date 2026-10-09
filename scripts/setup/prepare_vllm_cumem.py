"""Build an isolated vLLM 0.11.0 CuMem fix without modifying installed packages."""

import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig


SOURCE_HASHES = {
    "csrc/cumem_allocator.cpp": "a85f7ad6ab095a97af9fcfa56aea69b78c7e0d40e1662b7370c29d2407400fe6",
    "vllm/device_allocator/cumem.py": "aef71f5640d466bf9452b551ab3c5f70d9e9858f38105aae4907cac1941e0542",
    "vllm/v1/worker/gpu_worker.py": "e06833e8a5247dea488f067078b9f64e1284ca9c97b248b279852b046e815256",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unmodified vLLM v0.11.0 source checkout.")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the ParaAgent checkout.")
    parser.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error("Keep the generated overlay outside the ParaAgent checkout")
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    versions = {name: metadata.version(name) for name in ("vllm", "torch")}
    if versions["vllm"] != "0.11.0" or versions["torch"].split("+")[0] != "2.8.0":
        parser.error(f"This patch is limited to vLLM 0.11.0 / torch 2.8.0, found {versions}")
    package = Path(util.find_spec("vllm").origin).parent
    for relative, expected in SOURCE_HASHES.items():
        source = args.source / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            parser.error(f"Unexpected source revision: {source}")
        if relative.startswith("vllm/"):
            installed = package / Path(relative).relative_to("vllm")
            if installed.read_bytes() != source.read_bytes():
                parser.error(f"Installed file already differs from the pinned source: {installed}")
    patch = root / "patches/vllm-0.11.0-cumem.patch"
    staging = output / "patched-source"
    for relative in SOURCE_HASHES:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / relative, target)
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=staging, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=staging, check=True)
    library = output / "cumem_allocator.abi3.so"
    command = [
        os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-shared", "-fPIC",
        "-DPy_LIMITED_API=0x03090000", f"-I{sysconfig.get_path('include')}",
        f"-I{args.cuda_home / 'include'}", str(staging / "csrc/cumem_allocator.cpp"),
        f"-L{args.cuda_home / 'lib64/stubs'}", "-lcuda", "-o", str(library),
    ]
    subprocess.run(command, check=True)
    overlay = output / "vllm"
    shutil.copytree(package, overlay,
                    copy_function=lambda source, target: Path(target).symlink_to(Path(source).resolve()),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in SOURCE_HASHES:
        if relative.startswith("vllm/"):
            target = output / relative
            target.unlink()
            shutil.copy2(staging / relative, target)
    for target in overlay.glob("cumem_allocator*.so"):
        target.unlink()
    library.rename(overlay / library.name)
    record = {"versions": versions, "base_package": str(package), "source_hashes": SOURCE_HASHES,
              "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
              "library_sha256": hashlib.sha256((overlay / library.name).read_bytes()).hexdigest(),
              "compile_command": command}
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Built isolated overlay: {output}\nPrepend this directory to PYTHONPATH in the RL environment.")


if __name__ == "__main__":
    main()
