"""Download and extract the official resources on Windows, macOS, or Linux."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile
import hashlib
import json

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ['HF_HOME'] = str(root / '.cache' / 'huggingface')
hf = Path(sys.executable).parent / ('hf.exe' if os.name == 'nt' else 'hf')
manifest = json.loads((root / 'resources.json').read_text(encoding='utf-8'))
for repo, destination in [
    ('SprintML/hackathon', 'starter-kit/setup'),
    ('SprintML/Hackathon_Watermark_Localisation_Task', 'starter-kit/task-1'),
]:
    command = [str(hf), 'download', repo, '--repo-type', 'dataset', '--local-dir', destination]
    if repo == manifest['dataset_repository']:
        command += ['--revision', manifest['revision']]
    subprocess.run(command, check=True)
kit = root / 'starter-kit/task-1'
for entry in manifest['files']:
    if 'lfs' in entry:
        archive = kit / entry['path']
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['lfs']['oid']:
            raise ValueError(f"Checksum mismatch: {archive}")
        print(f"SHA-256 verified: {archive.name}", flush=True)
with zipfile.ZipFile(kit / 'starter_kit.zip') as z:
    z.extractall(kit)
for name in ['train.zip', 'test.zip']:
    with zipfile.ZipFile(kit / name) as z:
        z.extractall(root / 'data')
for name in ['wmloc', 'download_models.py', 'submission_template.py']:
    dst = root / "official" / name if name == "submission_template.py" else root / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if (kit / name).is_dir():
            shutil.copytree(kit / name, dst)
        else:
            shutil.copy2(kit / name, dst)
