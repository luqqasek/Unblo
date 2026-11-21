"""Download the three watermark models into ./models (or --models_dir).
  - PixelSeal: clones github.com/facebookresearch/videoseal at a pinned commit and downloads the PixelSeal
    checkpoint; writes models/pixelseal_card.yaml with absolute paths.
  - TrustMark Q: prefetches weights into the installed package and checks its published checksums.
  - StegaStamp: downloads the WAVES ONNX export and writes a copy that also outputs the decoder logits.
Needs: git, `pip install onnx` (only for the StegaStamp step)."""
import argparse, subprocess, urllib.request
import shutil
import sys
import hashlib
import importlib
from pathlib import Path

VIDEOSEAL_COMMIT = "b1bf4c0"
PIXELSEAL_CKPT = "https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth"
STEGASTAMP_ONNX = "https://github.com/umd-huang-lab/WAVES/raw/main/decoders/stega_stamp.onnx"


def fetch(url, dst: Path, expected_md5=None):
    valid = dst.exists() and dst.stat().st_size > 0
    if valid and expected_md5:
        with dst.open("rb") as stream:
            valid = hashlib.file_digest(stream, "md5").hexdigest() == expected_md5
    if valid:
        print("exists:", dst); return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url, flush=True)
    partial = dst.with_suffix(".part")
    # Windows curl uses the native network stack and has explicit retry/timeouts.
    if sys.platform == "win32" and shutil.which("curl.exe"):
        subprocess.run(["curl.exe", "-L", "--fail", "--retry", "3", "--connect-timeout", "20",
                        "--max-time", "900", "-o", str(partial), url], check=True)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if expected_md5:
        with partial.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() != expected_md5:
                raise ValueError(f"Checksum mismatch: {dst.name}")
    partial.replace(dst)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models_dir", default=str(Path(__file__).resolve().parent / "models"))
    d = Path(ap.parse_args().models_dir).resolve(); d.mkdir(parents=True, exist_ok=True)
    repo = d / "videoseal"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/facebookresearch/videoseal.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", VIDEOSEAL_COMMIT], check=True)
    fetch(PIXELSEAL_CKPT, d / "pixelseal" / "checkpoint.pth")
    card = (repo / "videoseal" / "cards" / "pixelseal.yaml").read_text()
    card = card.replace(PIXELSEAL_CKPT, str(d / "pixelseal" / "checkpoint.pth"))
    card = card.replace("attenuation_config: configs/", f"attenuation_config: {repo}/configs/")
    (d / "pixelseal_card.yaml").write_text(card)
    fetch(STEGASTAMP_ONNX, d / "stegastamp" / "stega_stamp.onnx")
    out = d / "stegastamp" / "stega_stamp_logits.onnx"
    if not out.exists():
        import onnx
        m = onnx.load(str(d / "stegastamp" / "stega_stamp.onnx"))
        m.graph.output.append(onnx.helper.make_tensor_value_info("gen_decoder/decoder_out/BiasAdd:0", onnx.TensorProto.FLOAT, None))
        onnx.save(m, str(out))
    # Prefetch via the same bounded downloader instead of TrustMark's urlretrieve.
    tm = importlib.import_module("trustmark.trustmark")
    tm_models = Path(tm.__file__).parent / "models"
    for name in ("trustmark_Q.yaml", "decoder_Q.ckpt", "encoder_Q.ckpt",
                 "trustmark_rm_Q.yaml", "trustmark_rm_Q.ckpt"):
        fetch(tm.MODEL_REMOTE_HOST + name, tm_models / name, tm.MODEL_CHECKSUMS[name])
    print("models ready in", d)


if __name__ == "__main__":
    main()
