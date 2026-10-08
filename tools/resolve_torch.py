# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve PyTorch-owned wheels once for all stages of a CUDA image build."""

import argparse
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlsplit

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename
from use_existing_torch import TORCH_LIBRARIES


def torch_requirements(path: Path) -> list[Requirement]:
    requirements = []
    for line in path.read_text().splitlines():
        line = line.partition("#")[0].strip()
        if line and not line.startswith("-"):
            requirement = Requirement(line)
            if requirement.name.lower() in TORCH_LIBRARIES:
                requirements.append(requirement)
    return requirements


def resolve(args: argparse.Namespace, cuda: str) -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    requirements = torch_requirements(args.requirements)
    if args.rubin_requirements:
        rubin = torch_requirements(args.rubin_requirements)
        replaced = {r.name for r in rubin}
        requirements = [r for r in requirements if r.name not in replaced] + rubin
    elif args.nightly:
        for requirement in requirements:
            requirement.specifier = SpecifierSet(
                ",".join(str(s) for s in requirement.specifier if s.operator != "==")
            )
    if {r.name for r in requirements} != set(TORCH_LIBRARIES):
        raise ValueError("CUDA requirements must declare every PyTorch-owned package")

    nightly = args.nightly or args.rubin_requirements is not None
    index = f"{args.index_base_url.rstrip('/')}/"
    index += f"{'nightly/' if nightly else ''}cu{cuda.replace('.', '')}"
    env = os.environ.copy()
    normal_indexes = shlex.split(env.pop("UV_INDEX", ""))
    normal_indexes += shlex.split(env.pop("UV_EXTRA_INDEX_URL", ""))
    normal_indexes.append(
        env.pop("UV_DEFAULT_INDEX", "")
        or env.get("UV_INDEX_URL")
        or "https://pypi.org/simple"
    )
    env.pop("UV_INDEX_URL", None)

    project = [
        "[project]",
        'name = "vllm-torch-build"',
        'version = "0"',
        f"dependencies = {json.dumps([str(r) for r in requirements])}",
        "[tool.uv.sources]",
        *(f'{name} = {{ index = "pytorch" }}' for name in TORCH_LIBRARIES),
    ]
    for normal_index in normal_indexes:
        project.append("[[tool.uv.index]]")
        # uv accepts both URL and name=URL index arguments.
        if "=" in normal_index.split(":", 1)[0]:
            name, normal_index = normal_index.split("=", 1)
            project.append(f"name = {json.dumps(name)}")
        project.append(f"url = {json.dumps(normal_index)}")
    project += [
        "[[tool.uv.index]]",
        'name = "pytorch"',
        f"url = {json.dumps(index)}",
        "default = true",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text("\n".join(project) + "\n")
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(root / "pyproject.toml"),
                "--python",
                sys.executable,
                *(
                    ["--python-platform", args.python_platform]
                    if args.python_platform
                    else []
                ),
                "--index-strategy",
                "unsafe-first-match",
                "--only-binary",
                ",".join(TORCH_LIBRARIES),
                "--prerelease",
                "allow" if nightly else "if-necessary",
                "--format",
                "pylock.toml",
                "--no-header",
                "--output-file",
                str(root / "pylock.toml"),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        lock = tomllib.loads((root / "pylock.toml").read_text())
    constraints = []
    for package in lock["packages"]:
        if package["name"] not in TORCH_LIBRARIES:
            continue
        wheels = package.get("wheels", [])
        if len(wheels) != 1:
            raise ValueError(f"Expected one target wheel for {package['name']}")
        wheel = wheels[0]
        url = wheel["url"]
        if sha256 := wheel.get("hashes", {}).get("sha256"):
            url = f"{urldefrag(url)[0]}#sha256={sha256}"
        constraints.append(f"{package['name']} @ {url}")
    if len(constraints) != len(TORCH_LIBRARIES):
        raise ValueError("The resolution omitted a PyTorch-owned package")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "constraints.txt").write_text("\n".join(constraints) + "\n")
    (args.output_dir / "index.txt").write_text(index + "\n")


def check(args: argparse.Namespace, cuda: str) -> None:
    import torch

    for line in (args.output_dir / "constraints.txt").read_text().splitlines():
        requirement = Requirement(line)
        url = requirement.url
        assert url is not None
        _, version, _, _ = parse_wheel_filename(
            unquote(urlsplit(url).path).rsplit("/", 1)[1]
        )
        installed = importlib.metadata.distribution(requirement.name)
        origin = json.loads(installed.read_text("direct_url.json") or "{}")
        if installed.version != str(version) or unquote(
            origin.get("url", "")
        ) != unquote(urldefrag(url)[0]):
            raise ValueError(
                f"{requirement.name} differs from the selected build wheel"
            )
    nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
    if f"release {cuda}," not in nvcc or torch.version.cuda != cuda:
        raise ValueError(
            f"CUDA {cuda} must match nvcc and torch.version.cuda={torch.version.cuda}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument(
        "--requirements", type=Path, default=Path("requirements/cuda.txt")
    )
    parser.add_argument("--rubin-requirements", type=Path)
    parser.add_argument("--nightly", action="store_true")
    parser.add_argument(
        "--python-platform", help="Override uv's native target platform"
    )
    parser.add_argument("--index-base-url", default="https://download.pytorch.org/whl")
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/torch-build"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    parts = args.cuda_version.split(".")
    if len(parts) not in (2, 3) or not all(
        p.isascii() and p.isdecimal() for p in parts
    ):
        parser.error("--cuda-version must be major.minor[.patch]")
    cuda = ".".join(parts[:2])
    (check if args.check else resolve)(args, cuda)


if __name__ == "__main__":
    main()
