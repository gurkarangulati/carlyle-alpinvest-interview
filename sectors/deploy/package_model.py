"""Pack the fine-tuned model folder into model.tar.gz, the format SageMaker loads.

Run:  uv run python sectors/deploy/package_model.py
"""

import tarfile
from pathlib import Path

SECTORS = Path(__file__).resolve().parents[1]
MODEL_DIR = SECTORS / "out" / "models" / "finetune_full"
TARBALL = SECTORS / "out" / "model.tar.gz"

with tarfile.open(TARBALL, "w:gz") as tar:
    for f in sorted(MODEL_DIR.iterdir()):
        tar.add(f, arcname=f.name)  # files at the archive's root, not inside a folder
        print(f"added {f.name:28} {f.stat().st_size / 1e6:7.1f} MB")
print(f"\n{TARBALL} ({TARBALL.stat().st_size / 1e6:.0f} MB compressed)")
