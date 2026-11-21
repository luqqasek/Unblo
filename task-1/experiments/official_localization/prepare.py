"""Fetch pinned detector source and weights; run using task-1/.venv Python."""
import hashlib
import importlib
from pathlib import Path
import shutil
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
TM_COMMIT = "4c60b9fdb4125fca3001f0803e6890a6b55371ac"
MIRROR_COMMIT = "d95ea307c73410c96557200656d2e0a59574eedb"
STEGA_HASHES = {
    "saved_model.pb": "f5526a12f0caf3bedb637ced8bf270f7d833a10de8c0858d64e1248c02c962b0",
    "variables/variables.data-00000-of-00001": "a608276fce5487fc59e4a79bb38ccf21936e6993107170c4e74dd2399f435069",
    "variables/variables.index": "7b60f3b7d14537a81b7b58b00053b1a37284962cbb74dcb1c83bf7159fc00640",
}


def checked_download(url, path, expected, algorithm):
    def valid(candidate):
        if not candidate.is_file():
            return False
        with candidate.open("rb") as stream:
            return hashlib.file_digest(stream, algorithm).hexdigest() == expected
    if not valid(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        subprocess.run(["curl.exe" if sys.platform == "win32" else "curl", "-sS", "-L", "--fail",
                        "--retry", "2", "--connect-timeout", "20", "--max-time", "600",
                        "-o", str(part), url], check=True)
        if not valid(part):
            raise ValueError(f"Invalid {algorithm}: {path.name}")
        part.replace(path)
    print(f"Verified {algorithm}: {path.name}", flush=True)


def main():
    repo = TASK / "checkpoints/trustmark-localization"
    if not repo.exists():
        subprocess.run(["git", "clone", "https://github.com/adobe/trustmark.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", TM_COMMIT], check=True)
    # Reuse the installed original Q embedder/decoder without upgrading the main environment.
    import trustmark
    installed_models = Path(trustmark.__file__).parent / "models"
    models = repo / "python/trustmark/models"
    models.mkdir(parents=True, exist_ok=True)
    for name in ("trustmark_Q.yaml", "encoder_Q.ckpt", "decoder_Q.ckpt"):
        if not (models / name).exists() and (installed_models / name).exists():
            shutil.copyfile(installed_models / name, models / name)
    # Import a separate module namespace only to read published resource metadata.
    spec = importlib.util.spec_from_file_location("trustmark.localization_resources", repo / "python/trustmark/trustmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("trustmark_Q.yaml", "encoder_Q.ckpt", "decoder_Q.ckpt", "trustmark_bbox_Q.yaml", "trustmark_bbox_Q.ckpt"):
        checked_download(module.MODEL_REMOTE_HOST + name, models / name, module.MODEL_CHECKSUMS[name], "md5")
    destination = TASK / "checkpoints/stegastamp-detector/detector_models/stegastamp_detector"
    print("StegaStamp: community mirror; original author weight identity unverified", flush=True)
    for name, digest in STEGA_HASHES.items():
        checked_download(f"https://media.githubusercontent.com/media/Gu-f/StegaStamp/{MIRROR_COMMIT}/detector_models/stegastamp_detector/{name}",
                         destination / name, digest, "sha256")


if __name__ == "__main__":
    main()
