"""Extract only named model/reference files from the user-supplied Cherry archive."""
import argparse
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
staging = root / "artifacts/cherry-keycaps"
staging.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.archive) as archive:
    for name, output in [
        ("Cherry Keycaps Full.fbx", "full.fbx"),
        ("LICENSE", "LICENSE"),
        ("README.md", "README.md"),
        ("screenshots/top.jpg", "top.jpg"),
        ("screenshots/detail.jpg", "detail.jpg"),
    ]:
        (staging / output).write_bytes(archive.read("cherry-mx-keycaps-main/" + name))
print(staging)
