from pathlib import Path
import struct
import hashlib
from PIL import Image


MAGIC = b"NVC360\0\0"  # 8 bytes
MODEL_IDS = {
    0: "dcvchem",
    1: "dcvchem360",
}
MODEL_NAME_TO_ID = {v: k for k, v in MODEL_IDS.items()}
PROJECTION_IDS = {
    0: "none",
    1: "erp",
}
PROJECTION_NAME_TO_ID = {v: k for k, v in PROJECTION_IDS.items()}
WEIGHTS_BASEFOLDER = Path(__file__).parent.parent / "model_weights"


def hash_weights(path: Path, length: int = 8) -> bytes:
    """Return truncated SHA256 digest of weights file as bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.digest()[:length]


def find_weights(weights_hash, label=''):
    for file in WEIGHTS_BASEFOLDER.iterdir():
        if file.is_file():
            file_hash = hash_weights(file)
            if file_hash == weights_hash:
                return file
    raise FileNotFoundError(f"Cannot find {label} weights used for encoding in the weights folder ({WEIGHTS_BASEFOLDER}). Required weights hash: '{weights_hash}'.")


def get_png_info(folder: Path):
    """
    Return (num_frames, width, height, bitdepth) for a folder of PNGs.
    Uses the first PNG to probe dimensions/bitdepth.
    """
    files = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() == ".png"])
    if not files:
        raise ValueError(f"No .png files found in {folder}")

    num_frames = len(files)

    with Image.open(files[0]) as im:
        width, height = im.size
        # For PNG, bit depth is accessible in .info or mode
        bitdepth = im.bits if hasattr(im, "bits") else 8  # fallback

    return num_frames, width, height, bitdepth


def write_header(f, frames, width, height, bitdepth, projection_id, quality, gop, model_id, intra_weights_hash: bytes, inter_weights_hash: bytes):
    f.write(MAGIC)
    f.write(struct.pack(">HHHBBBHB", frames, width, height, bitdepth, projection_id, quality, gop, model_id))
    f.write(intra_weights_hash)
    f.write(inter_weights_hash)


def read_header(f):
    start = f.tell()
    magic = f.read(8)
    if magic != MAGIC:
        raise ValueError("Invalid magic")
    frames, width, height, bitdepth, projection_id, quality, gop, model_id = struct.unpack(">HHHBBBHB", f.read(12))
    intra_weights_hash = f.read(8)
    inter_weights_hash = f.read(8)
    end = f.tell()
    header_size = end - start
    return {
        "frames": frames,
        "width": width,
        "height": height,
        "bitdepth": bitdepth,
        "projection_id": projection_id,
        "quality": quality,
        "gop": gop,
        "model_id": model_id,
        "intra_weights_hash": intra_weights_hash,
        "inter_weights_hash": inter_weights_hash
    }, header_size
