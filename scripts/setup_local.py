import os
import subprocess
import sys
from pathlib import Path

root = Path(os.environ.get("DATA_REPO_ROOT", ""))
print(f"DATA_REPO_ROOT: {root}")

if not root:
    print("❌ DATA_REPO_ROOT is not set")
    sys.exit(1)

if not root.exists():
    print(f"❌ Data repo not found at {root}")
    sys.exit(1)

print("🔧 Adding local data-quarry")
subprocess.check_call(
    [
        "uv",
        "pip",
        "install",
        "-e",
        str(root),
    ]
)


print("✅ Local dependency added")
