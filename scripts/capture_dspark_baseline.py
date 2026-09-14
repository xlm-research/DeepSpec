"""Archive the actual DeepSpec working tree before draft training changes.

Run with the existing training interpreter. The destination must not exist.
Large model weights and compiled dependencies are identified by path/version;
fixed features, initial draft weights and update tensors are emitted separately
by tests.test_dspark_training_baseline.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def capture(destination: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    destination.mkdir(parents=True, exist_ok=False)
    files: list[Path] = []
    for directory in (
        "deepspec",
        "tests",
        "config",
        "scripts",
        "docs",
        "doc/specs",
        ".scratch/dspark-draft-torchtitan",
    ):
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    for name in (
        "train.py",
        "requirements.txt",
        "requirements-deepep.txt",
        "NOTICE",
        "CONTEXT.md",
        ".gitignore",
    ):
        if (root / name).is_file():
            files.append(root / name)
    hashes = {}
    with tarfile.open(destination / "working-tree.tar.gz", "w:gz") as archive:
        for path in sorted(files):
            name = str(path.relative_to(root))
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name, recursive=False)
    (destination / "files.sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for filename, arguments in (
        ("git-status.txt", ["status", "--short"]),
        ("git-diff.patch", ["diff", "--binary", "HEAD"]),
        ("head.txt", ["rev-parse", "HEAD"]),
    ):
        (destination / filename).write_bytes(
            subprocess.check_output(["git", *arguments], cwd=root)
        )
    dependencies: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "vllm",
        "torchtitan",
        "spmd_types",
        "torch_remat",
        "tyro",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    checkouts = {}
    for name in ("vllm", "torchtitan"):
        checkout = root / name
        if (checkout / ".git").exists():
            checkouts[name] = {
                "path": str(checkout),
                "head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
                ).strip(),
                "status": subprocess.check_output(
                    ["git", "status", "--short"], cwd=checkout, text=True
                ),
            }
    (destination / "environment.json").write_text(
        json.dumps(
            {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "executable": sys.executable,
                "python": sys.version,
                "dependencies": dependencies,
                "checkouts": checkouts,
            },
            indent=2,
        )
        + "\n"
    )
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    capture(parser.parse_args().destination.resolve())
