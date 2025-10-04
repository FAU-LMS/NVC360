from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO
import torch
import torch.nn.functional as F
from PIL import Image
import struct
import numpy as np


class PNGReader:
    def __init__(self, folder: Path, bitdepth: int = 8):
        self.files = sorted(Path(folder).glob("*.png"))
        self.num_frames = len(self.files)
        self.bitdepth = bitdepth

    def read(self, i: int) -> torch.Tensor:
        img = Image.open(self.files[i]).convert("RGB")
        arr = np.array(img)
        tensor = torch.from_numpy(arr).to(torch.float32)
        tensor = tensor.permute(2, 0, 1) / float((1 << self.bitdepth) - 1)
        return tensor.unsqueeze(0)


class PNGWriter:
    def __init__(self, folder: Path, bitdepth: int = 8):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.bitdepth = bitdepth

    def write(self, tensor: torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]  # drop batch
        arr = (tensor.clamp(0, 1) * ((1 << self.bitdepth) - 1)) \
              .permute(1, 2, 0).detach().cpu().numpy()
        arr = arr.astype(np.uint16 if self.bitdepth > 8 else np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        filename = self.folder / f"frame{self.counter:03d}.png"
        img.save(filename, "PNG", bits=self.bitdepth)
        self.counter += 1
        return filename


class CompressionModelController(ABC):

    @abstractmethod
    def encode(self,
               input_path: Path,
               bitstream: BinaryIO,
               width: int,
               height: int,
               bitdepth: int,
               frames: int):
        pass

    @abstractmethod
    def decode(self,
               bitstream: BinaryIO,
               output_path: Path,
               width: int,
               height: int,
               bitdepth: int,
               frames: int):
        pass

    @staticmethod
    def pad(x, min_div):
        height, width = x.size(2), x.size(3)
        new_h = (height + min_div - 1) // min_div * min_div
        new_w = (width + min_div - 1) // min_div * min_div
        padding_r = new_w - width
        padding_b = new_h - height
        return F.pad(x, (0, padding_r, 0, padding_b), mode="replicate")

    @staticmethod
    def crop(x, size):
        H, W = x.size(2), x.size(3)
        h, w = size
        padding_r = W - w
        padding_b = H - h
        return F.pad(x, (0, -padding_r, 0, -padding_b))

    @classmethod
    def write_uints(cls, fd, values, fmt="<{:d}I"):
        fd.write(struct.pack(fmt.format(len(values)), *values))
        return len(values) * 4

    @classmethod
    def write_uchars(cls, fd, values, fmt="<{:d}B"):
        fd.write(struct.pack(fmt.format(len(values)), *values))
        return len(values) * 1

    @classmethod
    def read_uints(cls, fd, n, fmt="<{:d}I"):
        sz = struct.calcsize("I")
        return struct.unpack(fmt.format(n), fd.read(n * sz))

    @classmethod
    def read_uchars(cls, fd, n, fmt="<{:d}B"):
        sz = struct.calcsize("B")
        return struct.unpack(fmt.format(n), fd.read(n * sz))

    @classmethod
    def write_bytes(cls, fd, values, fmt="<{:d}s"):
        if len(values) == 0:
            return
        fd.write(struct.pack(fmt.format(len(values)), values))
        return len(values) * 1

    @classmethod
    def read_bytes(cls, fd, n, fmt="<{:d}s"):
        sz = struct.calcsize("s")
        return struct.unpack(fmt.format(n), fd.read(n * sz))[0]
