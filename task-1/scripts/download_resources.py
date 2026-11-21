"""Download and extract the official resources on Windows, macOS, or Linux."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ['HF_HOME'] = str(root / '.cache' / 'huggingface')
hf = Path(sys.executable).parent / ('hf.exe' if os.name == 'nt' else 'hf')
for repo, destination in [
    ('SprintML/hackathon', 'starter-kit/setup'),
    ('SprintML/Hackathon_Watermark_Localisation_Task', 'starter-kit/task-1'),
]:
    subprocess.run([str(hf), 'download', repo, '--repo-type', 'dataset', '--local-dir', destination], check=True)
kit = root / 'starter-kit/task-1'
with zipfile.ZipFile(kit / 'starter_kit.zip') as z:
    z.extractall(kit)
for name in ['train.zip', 'test.zip']:
    with zipfile.ZipFile(kit / name) as z:
        z.extractall(root / 'data')
for name in ['wmloc', 'download_models.py', 'submission_template.py']:
    dst = root / name
    if not dst.exists():
        if (kit / name).is_dir():
            shutil.copytree(kit / name, dst)
        else:
            shutil.copy2(kit / name, dst)
