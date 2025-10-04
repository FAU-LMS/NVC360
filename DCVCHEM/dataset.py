import random
import warnings
from typing import Union, Tuple
from pathlib import Path
import pandas as pd
import yuvio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
from torch.nn.functional import grid_sample
import numpy as np
from skimage.transform import resize
import json
import re


class Dataset360(Dataset):
    def __init__(self,
                 dataset_dir: Union[str, Path],
                 sequence_length,
                 filter_license=None,
                 patch_size=(256, 256),
                 flow_threshold=0.8):
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.flow_threshold = flow_threshold
        self.df = pd.read_csv(self.dataset_dir / "dataset360.csv", index_col=[0])
        if filter_license is not None:
            self.df = self.df[self.df['license'].isin(filter_license)]
        self.drop_below_flow_threshold()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        clip = self.df.iloc[idx]
        clip_path = self.dataset_dir / str(clip['video_id']) / f"{clip['clip_id']:02d}"

        with torch.no_grad():
            image_paths = [str(clip_path / f"{i+1:02d}.png") for i in range(self.sequence_length)]
            image_sequence = map(Image.open, image_paths)
            image_sequence = list(map(to_tensor, image_sequence))
            image_sequence = torch.stack(image_sequence, dim=0)  # [f, c, h, w]
            f, c, h, w = image_sequence.shape

            # Sample patch parameters
            flow = np.load(clip_path / "flow.npy")
            mask = flow >= self.flow_threshold
            src_patch_location = self.sample_patch_location((h, w), mask)
            tar_patch_location = self.sample_patch_location((h, w))

            # Prepare warp grid
            xmin, ymin, xmax, ymax = self.get_patch_bounding_box(tar_patch_location, self.patch_size)
            x = torch.arange(xmin, xmax)
            y = torch.arange(ymin, ymax)
            pos = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
            pos_sphere = self.erp_to_sphere(pos, (h, w))  # [h, w, 3]

            # Rotate warp grid
            rotation_matrix = self.patch_rotation_matrix(src_patch_location,
                                                         tar_patch_location,
                                                         (h, w))
            pos_sphere = torch.tensordot(pos_sphere, rotation_matrix, dims=([2], [1]))

            # Get warped grid and normalize
            pos_warp = self.erp_from_sphere(pos_sphere, (h, w))
            pos_warp[..., 0] = (pos_warp[..., 0] / (w / 2)) - 1
            pos_warp[..., 1] = (pos_warp[..., 1] / (h / 2)) - 1
            pos[..., 0] = (pos[..., 0] / (w / 2)) - 1
            pos[..., 1] = (pos[..., 1] / (h / 2)) - 1

            # Warp image sequence to target patch
            image_sequence = grid_sample(image_sequence,
                                         pos_warp.unsqueeze(0).expand(f, -1, -1, -1),
                                         mode='bicubic',
                                         padding_mode='reflection',
                                         align_corners=False)
            torch.clamp_(image_sequence, 0, 1)

        return image_sequence, pos

    def drop_below_flow_threshold(self):
        """
        Drops clips from the dataset that do not include sufficient motion according
        to the defined flow threshold.
        Includes a caching mechanism (persisted at `~/.dataset360/config.cache`) to
        avoid recalculation each run.
        """
        cache_dir = Path.home() / ".dataset360"
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file = cache_dir / "config.cache"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = dict()
        cache_key = f"{self.flow_threshold:.10f}"
        if cache_key in cache:
            drop_idxs = list(map(int, cache[cache_key].split(",")))
        else:
            drop_idxs = []
            for idx, clip in self.df.iterrows():
                clip_path = self.dataset_dir / str(clip['video_id']) / f"{clip['clip_id']:02d}"
                flow = np.load(clip_path / "flow.npy")
                mask = flow >= self.flow_threshold
                mask = resize(mask, (clip['height'], clip['width']), order=0)
                if np.count_nonzero(mask) == 0:
                    drop_idxs.append(idx)
            cache[cache_key] = ",".join(map(str, drop_idxs))
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
                f.write("\n")
        if len(drop_idxs) >= 0:
            self.df.drop(index=drop_idxs, inplace=True)
            warnings.warn(f"Dropped {len(drop_idxs)} clips due to motion below flow threshold.")


    @classmethod
    def unitsphere_to_cartesian(cls, input: torch.Tensor):
        sin_theta = torch.sin(input[..., 0])
        x = sin_theta * torch.cos(input[..., 1])
        y = sin_theta * torch.sin(input[..., 1])
        z = torch.cos(input[..., 0])
        return torch.stack((x, y, z), dim=-1)

    @classmethod
    def cartesian_to_unitsphere(cls, input: torch.Tensor):
        r = torch.sqrt(torch.sum(torch.square(input), dim=-1))
        theta = torch.acos(input[..., 2] / r)
        phi = torch.atan2(input[..., 1], input[..., 0])
        return torch.stack((theta, phi), dim=-1)

    @classmethod
    def erp_to_sphere(cls, pos: torch.Tensor, size: Tuple[int, int], mode='cartesian'):
        """
        :param pos: Pixel positions (x, y) in erp domain [..., h, w, 2]
        :param size: Size of erp image in pixels (height, width)
        :param mode: Output mode 'cartesian' (x, y, z) or 'spherical' (theta, phi)
        """
        theta = ((pos[..., 1] + 0.5) / size[0]) * torch.pi
        phi = -((pos[..., 0] + 0.5) / size[1]) * 2 * torch.pi
        spherical = torch.stack((theta, phi), dim=-1)
        if mode == 'spherical':
            return spherical
        elif mode == 'cartesian':
            return cls.unitsphere_to_cartesian(spherical)
        else:
            raise ValueError(f"Unknown output mode '{mode}'.")

    @classmethod
    def erp_from_sphere(cls, pos: torch.Tensor, size: Tuple[int, int], mode='cartesian'):
        """
        :param pos: Pixel positions (x, y, z) on sphere [..., h, w, 3]
        :param size: Size of erp image in pixels (height, width)
        :param mode: Input mode 'cartesian' (x, y, z) or 'spherical' (theta, phi)
        """
        if mode == 'spherical':
            pass
        elif mode == 'cartesian':
            pos = cls.cartesian_to_unitsphere(pos)
        else:
            raise ValueError(f"Unknown input mode '{mode}'.")
        pos[..., 1] = torch.where(pos[..., 1] > 0, pos[..., 1] - 2 * torch.pi, pos[..., 1])
        x = -(pos[..., 1] / (2 * torch.pi)) * size[1] - 0.5
        y = (pos[..., 0] / torch.pi) * size[0] - 0.5
        return torch.stack((x, y), dim=-1)

    @classmethod
    def rotation_matrix_x(cls, angle, device=None):
        return torch.tensor([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], dtype=torch.float32, device=device)

    @classmethod
    def rotation_matrix_y(cls, angle, device=None):
        return torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=torch.float32, device=device)

    @classmethod
    def rotation_matrix_z(cls, angle, device=None):
        return torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

    @classmethod
    def patch_rotation_matrix(cls, src_patch_location, tar_patch_location, size, device=None):
        src_patch_sphere = cls.erp_to_sphere(torch.tensor(src_patch_location), size, mode='spherical')
        tar_patch_sphere = cls.erp_to_sphere(torch.tensor(tar_patch_location), size, mode='spherical')
        rotation_matrix_1 = cls.rotation_matrix_z(-tar_patch_sphere[..., 1], device=device)
        rotation_matrix_2 = cls.rotation_matrix_y(-(tar_patch_sphere[..., 0] - src_patch_sphere[..., 0]), device=device)
        rotation_matrix_3 = cls.rotation_matrix_z(src_patch_sphere[..., 1], device=device)
        rotation_matrix = torch.mm(rotation_matrix_3, torch.mm(rotation_matrix_2, rotation_matrix_1))
        return rotation_matrix

    @classmethod
    def sample_patch_location(cls, size, mask=None):
        if mask is None:
            mask = np.ones(size, dtype=bool)
        else:
            mask = resize(mask, size, order=0)
            # TODO: Recalculate flow to image size
            # if mask.shape != size:
            #     raise ValueError(f"Mask shape does not match specified size '{size}'")
        valid_patch_pos = np.argwhere(mask)
        patch_pos_idx = np.random.randint(0, valid_patch_pos.shape[0])
        patch_pos = valid_patch_pos[patch_pos_idx]
        return patch_pos[::-1].copy()

    @classmethod
    def get_patch_bounding_box(cls, patch_location, patch_size):
        return (patch_location[0] - patch_size[0] // 2,
                patch_location[1] - patch_size[1] // 2,
                patch_location[0] + patch_size[0] // 2,
                patch_location[1] + patch_size[1] // 2)


class Vimeo90kDataset(Dataset):
    def __init__(self, root, sequence_length, transform=None, split="train", tuplet=3):
        list_path = Path(root) / self._list_filename(split, tuplet)

        with open(list_path) as f:
            self.samples = [
                [f"{root}/sequences/{line.rstrip()}/im{i+1}.png" for i in range(sequence_length)]
                for line in f if line.strip() != ""
            ]

        self.transform = transform

    def __getitem__(self, index):
        image_paths = self.samples[index]
        image_sequence = map(Image.open, image_paths)
        image_sequence = list(map(self.transform, image_sequence))
        image_sequence = torch.stack(image_sequence, dim=0)  # [f, c, h, w]
        f, c, h, w = image_sequence.shape

        pos = torch.zeros((h, w, 2), dtype=torch.float32)

        return image_sequence, pos

    def __len__(self):
        return len(self.samples)

    def _list_filename(self, split: str, tuplet: int) -> str:
        tuplet_prefix = {3: "tri", 7: "sep"}[tuplet]
        list_suffix = {"train": "trainlist", "valid": "testlist"}[split]
        return f"{tuplet_prefix}_{list_suffix}.txt"


class JVET360Dataset(Dataset):

    class Sequence:
        class Meta:
            def __init__(self,
                         width: int,
                         height: int,
                         bitdepth: int,
                         chroma_format: str,
                         framerate: int,
                         frames: int):
                self.width = width
                self.height = height
                self.bitdepth = bitdepth
                self.chroma_format = chroma_format
                self.framerate = framerate
                self.frames = frames

            @classmethod
            def from_filename(cls, file, frames):
                name = Path(file).stem
                width = int(re.search(r"(?<=_)\d*(?=x)", name).group(0))
                height = int(re.search(r"(?<=x)\d*(?=_)", name).group(0))
                framerate = int(re.search(r"(?<=_)\d*(?=fps)", name).group(0))
                bitdepth = int(re.search(r"(?<=_)\d*(?=bit)", name).group(0))
                chroma_format = re.search(r"(?<=bit_)\d*", name).group(0)
                return cls(width, height, bitdepth, chroma_format, framerate, frames)

            def pixel_format(self):
                pixel_format = "yuv" + self.chroma_format + "p"
                if self.chroma_format == '400':
                    pixel_format = 'gray'
                if self.bitdepth != 8:
                    pixel_format += f"{self.bitdepth}le"
                return pixel_format

        def __init__(self, path, frames: int):
            self.path = Path(path)
            self.meta = self.Meta.from_filename(path, frames)

    def __init__(self, jvet360_basepath, sequence_length, patch_size):
        basepath = Path(jvet360_basepath)
        self.sequences = [
            self.Sequence(basepath / "SkateboardInLot_8192x4096_30fps_10bit_420_erp.yuv", 300),
            self.Sequence(basepath / "ChairliftRide_8192x4096_30fps_10bit_420_erp/"
                                     "ChairliftRide_8192x4096_30fps_10bit_420_erp.yuv", 300),
            self.Sequence(basepath / "KiteFlite_8192x4096_30fps_8bit_420_erp/"
                                     "KiteFlite_8192x4096_30fps_8bit_420_erp.yuv", 300),
            self.Sequence(basepath / "Harbor_8192x4096_30fps_8bit_420_erp/"
                                     "Harbor_8192x4096_30fps_8bit_420_erp.yuv", 300),
            self.Sequence(basepath / "Trolley_8192x4096_30fps_8bit_420_erp/"
                                     "Trolley_8192x4096_30fps_8bit_420_erp.yuv", 300),
            self.Sequence(basepath / "Gaslamp_8192x4096_30fps_8bit_420_erp.yuv", 300),
            self.Sequence(basepath / "Balboa_6144x3072_60fps_8bit_420_erp/"
                                     "Balboa_6144x3072_60fps_8bit_420_erp.yuv", 600),
            self.Sequence(basepath / "Broadway_6144x3072_60fps_8bit_420_erp/"
                                     "Broadway_6144x3072_60fps_8bit_420_erp.yuv", 600),
            self.Sequence(basepath / "Landing2_6144x3072_30fps_8bit_420_erp.yuv", 300),
            self.Sequence(basepath / "BranCastle2_6144x3072_30fps_8bit_420_erp/"
                                     "BranCastle2_6144x3072_30fps_8bit_420_erp.yuv", 300),
        ]
        self.sequence_length = sequence_length
        self.patch_size = patch_size

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        reader = yuvio.get_reader(
            sequence.path,
            sequence.meta.width,
            sequence.meta.height,
            sequence.meta.pixel_format()
        )
        max_val = float(2 ** sequence.meta.bitdepth - 1)
        image_sequence = []
        for i in range(self.sequence_length):
            yuv_frame = reader.read(i, 1)[0]
            y, u, v = yuv_frame.split()
            y = y.astype(float) / max_val
            u = u.astype(float) / max_val
            v = v.astype(float) / max_val
            y = np.round(resize(y, self.patch_size, order=3) * 255).astype(np.uint8)
            u = np.round(resize(u, self.patch_size, order=3) * 255).astype(np.uint8)
            v = np.round(resize(v, self.patch_size, order=3) * 255).astype(np.uint8)
            rgb = yuvio.to_rgb(
                yuvio.frame((y, u, v), "yuv444p"),
                'bt709',
                'limited'
            )
            rgb = rgb.astype(np.float32) / 255
            rgb = torch.from_numpy(rgb).movedim(2, 0)
            image_sequence.append(rgb)
        image_sequence = torch.stack(image_sequence, dim=0)

        x = torch.arange(0, self.patch_size[1])
        y = torch.arange(0, self.patch_size[0])
        pos = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        pos[..., 0] = (pos[..., 0] / (self.patch_size[1] / 2)) - 1
        pos[..., 1] = (pos[..., 1] / (self.patch_size[0] / 2)) - 1

        return image_sequence, pos

    def __len__(self):
        return len(self.sequences)


if __name__ == '__main__':
    # data_set = Dataset360(
    #     "/home/fa94ciqu/Resources/dataset360/clips",
    #     1
    # )
    data_set = JVET360Dataset(
        "/CLUSTERHOMES/LMS/sequences/jvet/360", 1, (200, 400)
    )
    for i, (image_sequence, pos) in enumerate(data_set):
        weights = torch.cos(pos[..., 1] * 0.5 * torch.pi)  # [b, h, w]
        sm = torch.sum(weights).item()
        n = pos.shape[0] * pos.shape[1] * pos.shape[2]
        print(sm, n, n/sm)

        image = to_pil_image(image_sequence[0])
        image.save(f"{i:02d}.png")


    # data_loader = DataLoader(data_set, batch_size=4, shuffle=True, num_workers=8)
    # for i, image_sequence in enumerate(data_loader):
    #     image_sequence = image_sequence.to('cuda:0')
    #     print(i, image_sequence.shape)
    #     del image_sequence
    # image_sequence = data_set[random.randint(0, len(data_set))]
    # grid = make_grid(image_sequence, 4)
    # to_pil_image(grid).save("grid.png")
