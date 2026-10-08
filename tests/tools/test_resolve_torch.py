# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise CUDA source ownership with uv and tiny offline wheel indexes."""

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
from use_existing_torch import TORCH_LIBRARIES  # noqa: E402


def wheel(index: Path, name: str, version: str, *metadata: str) -> None:
    package = index / name
    package.mkdir(parents=True, exist_ok=True)
    path = package / f"{name}-{version}-py3-none-any.whl"
    info = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{info}/METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.1",
                    f"Name: {name}",
                    f"Version: {version}",
                    *metadata,
                    "",
                ]
            ),
        )
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{info}/RECORD", "")
    (package / "index.html").write_text(
        "\n".join(
            f'<a href="{p.name}#sha256={hashlib.sha256(p.read_bytes()).hexdigest()}">'
            "wheel</a>"
            for p in sorted(package.glob("*.whl"))
        )
    )


@pytest.fixture
def indexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith(("UV_", "PIP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_INDEX_URL", (tmp_path / "normal").as_uri())
    for name in TORCH_LIBRARIES:
        wheel(tmp_path / "normal", name, "9.0")
        wheel(
            tmp_path / "selected/cu129", name, "1.0+cu129", "Requires-Dist: ordinary>=1"
        )
    wheel(tmp_path / "normal", "ordinary", "2.0")
    wheel(tmp_path / "selected/cu129", "ordinary", "3.0")
    (tmp_path / "cuda.txt").write_text(
        "\n".join(
            f"{name}>=1" if name == "torchcodec" else f"{name}==1.0"
            for name in TORCH_LIBRARIES
        )
    )
    return tmp_path


def resolve(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/resolve_torch.py"),
            "--cuda-version",
            "12.9.1",
            "--requirements",
            str(root / "cuda.txt"),
            "--index-base-url",
            (root / "selected").as_uri(),
            "--output-dir",
            str(root / "result"),
            *args,
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )


def test_bound_wheels_survive_later_installs(indexes: Path) -> None:
    """Newer generic wheels cannot escape the build's exact URL constraints."""
    result = resolve(indexes)
    assert result.returncode == 0, result.stderr
    constraints = indexes / "result/constraints.txt"
    text = constraints.read_text()
    assert text.count("#sha256=") == len(TORCH_LIBRARIES)
    assert "ordinary" not in text
    assert 'name = "ordinary"\nversion = "2.0"' in result.stdout
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(indexes / "venv")], check=True
    )
    install = ["uv", "pip", "install", "--python", str(indexes / "venv/bin/python")]
    env = dict(os.environ, UV_CONSTRAINT=str(constraints))
    subprocess.run([*install, "-r", str(constraints)], env=env, check=True)
    conflict = subprocess.run([*install, "torchcodec>=9"], env=env, capture_output=True)
    assert conflict.returncode != 0
    # Explicit -c replaces UV_CONSTRAINT; connector constraints must include it.
    connector_constraints = indexes / "connector-constraints.txt"
    connector_constraints.write_text(text + "ordinary==2.0\n")
    conflict = subprocess.run(
        [*install, "-c", str(connector_constraints), "torchcodec>=9"],
        env=env,
        capture_output=True,
    )
    assert conflict.returncode != 0
    # The constraints file alone does not change downstream installation behavior.
    subprocess.run([*install, "torchcodec>=9"], check=True)


@pytest.mark.parametrize("requirement", ["torchcodec>=9", "torchcodec>=1"])
def test_unavailable_selected_wheel_never_falls_back(
    indexes: Path, requirement: str
) -> None:
    if requirement.endswith(">=1"):
        (indexes / "selected/cu129/torchcodec/index.html").write_text("")
    path = indexes / "cuda.txt"
    path.write_text(path.read_text().replace("torchcodec>=1", requirement))
    result = resolve(indexes)
    assert result.returncode != 0
    assert not (indexes / "result/constraints.txt").exists()


def test_isolated_build_uses_selected_wheel(indexes: Path) -> None:
    """A build backend's dependencies must obey the same ownership policy."""
    result = resolve(indexes)
    assert result.returncode == 0, result.stderr
    source = indexes / "source"
    wheel(source, "example", "1.0")
    (source / "backend.py").write_text(
        "from pathlib import Path\nfrom shutil import copy2\n"
        "def build_wheel(wheel_directory, *args, **kwargs):\n"
        "    wheel = next(Path(__file__).parent.glob('example/*.whl'))\n"
        "    copy2(wheel, wheel_directory)\n    return wheel.name\n"
    )
    project = source / "pyproject.toml"
    project.write_text(
        '[build-system]\nrequires = ["torchcodec>=1"]\n'
        'build-backend = "backend"\nbackend-path = ["."]\n'
    )
    env = dict(os.environ, UV_BUILD_CONSTRAINT=str(indexes / "result/constraints.txt"))
    build = ["uv", "build", "--python", sys.executable, "--wheel", str(source)]
    result = subprocess.run(build, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    project.write_text(project.read_text().replace("torchcodec>=1", "torchcodec>=9"))
    result = subprocess.run(build, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "torchcodec>=9" in result.stderr


def test_nightly_resolves_transitives_and_keeps_ranges(indexes: Path) -> None:
    selected = indexes / "selected/nightly/cu129"
    for name in TORCH_LIBRARIES:
        wheel(selected, name, "1.1.dev1+cu129", "Requires-Dist: nightlycompiler==1")
        wheel(selected, name, "2.0.dev1+cu129", "Requires-Python: >=900")
    wheel(selected, "nightlycompiler", "1")
    wheel(indexes / "normal", "nightlycompiler", "2")
    result = resolve(indexes, "--nightly")
    assert result.returncode == 0, result.stderr
    assert "nightlycompiler" in result.stdout
    assert (indexes / "result/constraints.txt").read_text().count("1.1.dev1") == 4
    assert (indexes / "result/index.txt").read_text().strip() == selected.as_uri()
    path = indexes / "cuda.txt"
    path.write_text(path.read_text().replace("torchcodec>=1", "torchcodec>=9"))
    assert resolve(indexes, "--nightly").returncode != 0


@pytest.mark.parametrize("arch,version", [("x86_64", "1.1"), ("aarch64", "1.2")])
def test_rubin_pins_use_target_markers(indexes: Path, arch: str, version: str) -> None:
    selected = indexes / "selected/nightly/cu129"
    for name in TORCH_LIBRARIES:
        wheel(selected, name, "1.1.dev1+cu129")
    wheel(selected, "torch", "1.2.dev1+cu129")
    pins = indexes / "rubin.txt"
    pins.write_text(
        'torch==1.1.dev1+cu129; platform_machine == "x86_64"\n'
        'torch==1.2.dev1+cu129; platform_machine == "aarch64"\n'
        "torchvision==1.1.dev1+cu129\ntorchaudio==1.1.dev1+cu129\n"
    )
    result = resolve(
        indexes,
        "--rubin-requirements",
        str(pins),
        "--python-platform",
        f"{arch}-manylinux_2_28",
    )
    assert result.returncode == 0, result.stderr
    assert f"torch-{version}.dev1" in (indexes / "result/constraints.txt").read_text()
