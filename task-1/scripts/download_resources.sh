#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
HF=.venv/bin/hf
if [[ ! -x "$HF" ]]; then HF=hf; fi
"$HF" download SprintML/hackathon --repo-type dataset --local-dir starter-kit/setup
"$HF" download SprintML/Hackathon_Watermark_Localisation_Task --repo-type dataset --local-dir starter-kit/task-1
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil, zipfile
root = Path.cwd()
kit = root / 'starter-kit/task-1'
with zipfile.ZipFile(kit / 'starter_kit.zip') as z:
    z.extractall(kit)
for name in ['train.zip', 'test.zip']:
    with zipfile.ZipFile(kit / name) as z:
        z.extractall(root / 'data')
# Upstream code is preserved separately. Do not overwrite local edits on reruns.
for name in ['wmloc', 'download_models.py', 'submission_template.py']:
    dst = root / name
    if not dst.exists():
        if (kit / name).is_dir(): shutil.copytree(kit / name, dst)
        else: shutil.copy2(kit / name, dst)
PY
