import warnings
from typing import List, Union, Tuple
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import grid_sample
import numpy as np
from skimage.transform import resize
import json


class UGC360(Dataset):
    def __init__(self,
                 dataset_files: List[Union[str, Path]],
                 sequence_length,
                 filter_license=None,
                 patch_size=(256, 256),
                 resize_range: Union[int, Tuple[float, float]] = 512,
                 flow_threshold=0.5,
                 reproject=True,
                 mipmap_levels=8):
        self.dataset_files = [Path(file) for file in dataset_files]
        self.sequence_length = sequence_length
        self.filter_license = filter_license
        self.patch_size = patch_size
        self.resize_range = resize_range
        self.flow_threshold = flow_threshold
        self.reproject = reproject
        self.mipmap_levels = mipmap_levels
        self.setup_dataframe()

    def __len__(self):
        return len(self.df)

    def setup_dataframe(self):
        dfs = []
        for file in self.dataset_files:
            df = pd.read_csv(file, index_col=None)
            df['root'] = file.parent
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)
        if self.filter_license is not None:
            self.df = self.df[self.df['license'].isin(self.filter_license)]
        self.drop_below_flow_threshold()

    def update_sequence_length(self, new_length):
        self.sequence_length = new_length

    def __getitem__(self, idx):
        clip = self.df.iloc[idx]
        clip_path = clip['root'] / str(clip['video_id']) / f"{clip['clip_id']:02d}"

        with torch.no_grad():
            image_paths = [str(clip_path / f"{i + 1:02d}.png") for i in range(self.sequence_length)]
            image_sequence = map(Image.open, image_paths)
            image_sequence = list(map(to_tensor, image_sequence))
            image_sequence = torch.stack(image_sequence, dim=0)  # [f, c, h, w]
            f, c, h, w = image_sequence.shape

            # 360 padding (horizontal: circular, vertical: replicate)
            pad_width, pad_height = np.asarray(self.patch_size) // 2
            image_sequence = torch.nn.functional.pad(image_sequence, (pad_width, pad_width, 0, 0, 0, 0),
                                                     mode='circular')
            image_sequence = torch.nn.functional.pad(image_sequence, (0, 0, pad_height, pad_height, 0, 0),
                                                     mode='replicate')
            padded_height, padded_width = image_sequence.shape[2:4]

            # Set target scale
            if isinstance(self.resize_range, int):
                h_tar = self.resize_range
                w_tar = 2 * h_tar
            else:
                min_h = max(self.resize_range[0] * h, self.patch_size[0])
                max_h = max(self.resize_range[1] * h, self.patch_size[0])
                h_tar = np.random.randint(min_h, max_h + 1)
                w_tar = 2 * h_tar

            # Sample patch parameters
            flow = np.load(clip_path / "flow.npy")
            mask_src = flow >= self.flow_threshold
            if not self.reproject:
                # Avoid out-of-bounds src_patch_location
                h_mask, w_mask = mask_src.shape
                patch_size_in_src = np.ceil(np.asarray(self.patch_size) * np.array([w_mask/w_tar, h_mask/h_tar])).astype(int)
                mask_src[:patch_size_in_src[1]//2, :] = False
                mask_src[-patch_size_in_src[1]//2:, :] = False
                mask_src[:, :patch_size_in_src[0]//2] = False
                mask_src[:, -patch_size_in_src[0]//2:] = False
                # Workaround if the entire valid flow is out-of-bounds
                if not np.any(mask_src):
                    # Calculate distance to patch box, use valid flow position with minimum distance
                    pos_flow = np.transpose(np.nonzero(flow >= self.flow_threshold))
                    dist_y_t = np.clip((patch_size_in_src[1] // 2) - pos_flow[:, 0], 0, np.inf)
                    dist_y_b = np.clip(pos_flow[:, 0] - (patch_size_in_src[1] // 2), 0, np.inf)
                    dist_y = dist_y_t + dist_y_b
                    dist_x_l = np.clip((patch_size_in_src[0] // 2) - pos_flow[:, 1], 0, np.inf)
                    dist_x_r = np.clip(pos_flow[:, 1] - (patch_size_in_src[0] // 2), 0, np.inf)
                    dist_x = dist_x_l + dist_x_r
                    dist = np.sqrt(np.square(dist_y) + np.square(dist_x))
                    min_idx = np.argmin(dist)
                    mask_src[pos_flow[min_idx, 0], pos_flow[min_idx, 1]] = True

            src_patch_location = self.sample_src_patch_location((h, w), mask_src)

            if self.reproject:
                tar_patch_location = self.sample_target_patch_location((h_tar, w_tar),
                                                                       self.patch_size)
            else:
                tar_patch_location = np.round(src_patch_location * np.array([w_tar/w, h_tar/h]))

            # Prepare warp grid
            xmin, ymin, xmax, ymax = self.get_patch_bounding_box(tar_patch_location, self.patch_size)
            x = torch.arange(xmin, xmax, dtype=torch.float32)
            y = torch.arange(ymin, ymax, dtype=torch.float32)
            pos = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
            pos_sphere = self.erp_to_sphere(pos, (h_tar, w_tar))  # [h, w, 3]

            # Rotate warp grid
            if self.reproject:
                rotation_matrix = self.patch_rotation_matrix(src_patch_location,
                                                             tar_patch_location,
                                                             (h, w),
                                                             (h_tar, w_tar))
                pos_sphere = torch.tensordot(pos_sphere, rotation_matrix, dims=([2], [1]))

            # Get warped grid taking padding into account
            pos_warp = self.erp_from_sphere(pos_sphere, (h, w))
            pos_warp[..., 0] = pos_warp[..., 0] + pad_width
            pos_warp[..., 1] = pos_warp[..., 1] + pad_height

            if self.mipmap_levels > 0:
                image_sequence_mipmap = self.mipmap(image_sequence, levels=self.mipmap_levels)

                # Estimate gradient of source coordinates (texture) w.r.t. target coordinates (screen space)
                # for mipmap level (level of detail) selection
                pos_warp_dx = (pos_warp[:, 2:] - pos_warp[:, :-2]) / 2
                pos_warp_dy = (pos_warp[2:, :] - pos_warp[:-2, :]) / 2

                # Reduce 360-boundary artifacts
                pos_warp_dx[..., 0] = torch.where(pos_warp_dx[..., 0] > w / 4, 1, pos_warp_dx[..., 0])
                pos_warp_dx[..., 0] = torch.where(pos_warp_dx[..., 0] < - w / 4, 1, pos_warp_dx[..., 0])
                pos_warp_dy[..., 0] = torch.where(pos_warp_dy[..., 0] > w / 4, 1, pos_warp_dy[..., 0])
                pos_warp_dy[..., 0] = torch.where(pos_warp_dy[..., 0] < - w / 4, 1, pos_warp_dy[..., 0])

                pos_warp_dx = torch.nn.functional.pad(pos_warp_dx.unsqueeze(0), (0, 0, 1, 1, 0, 0),
                                                      mode='replicate').squeeze()
                pos_warp_dy = torch.nn.functional.pad(pos_warp_dy.unsqueeze(0), (0, 0, 0, 0, 1, 1),
                                                      mode='replicate').squeeze()

                # Select level of detail (LOD) based on estimated gradients
                pos_warp_du = torch.sqrt(torch.square(pos_warp_dx[:, :, 0]) + torch.square(pos_warp_dy[:, :, 0]))
                pos_warp_dv = torch.sqrt(torch.square(pos_warp_dx[:, :, 1]) + torch.square(pos_warp_dy[:, :, 1]))
                lod = torch.floor(torch.log2(torch.max(pos_warp_du, pos_warp_dv)))  # Floor LOD
                lod = torch.max(torch.full_like(lod, 0), torch.min(torch.full_like(lod, self.mipmap_levels), lod))

                lod_scale = torch.pow(2.0, -lod)
                pos_warp[..., 0] = torch.where(lod > 0, padded_width, 0) + pos_warp[..., 0] * lod_scale

                y_offset = torch.zeros_like(pos_warp[..., 1])
                level_offset = padded_height
                for level in range(2, self.mipmap_levels):
                    level_offset = level_offset // 2
                    y_offset = torch.where(lod >= level, y_offset + level_offset, y_offset)
                pos_warp[..., 1] = y_offset + pos_warp[..., 1] * lod_scale

            # Normalize
            if self.mipmap_levels > 0:
                pos_warp[..., 0] = (pos_warp[..., 0] / (image_sequence_mipmap.size(3) / 2)) - 1
                pos_warp[..., 1] = (pos_warp[..., 1] / (image_sequence_mipmap.size(2) / 2)) - 1
            else:
                pos_warp[..., 0] = (pos_warp[..., 0] / (image_sequence.size(3) / 2)) - 1
                pos_warp[..., 1] = (pos_warp[..., 1] / (image_sequence.size(2) / 2)) - 1

            # Warp image sequence to target patch
            image_sequence = grid_sample(image_sequence_mipmap if self.mipmap_levels > 0 else image_sequence,
                                         pos_warp.unsqueeze(0).expand(f, -1, -1, -1),
                                         mode='bilinear' if self.mipmap_levels > 0 else 'bicubic',
                                         padding_mode='zeros',
                                         align_corners=False)

            torch.clamp_(image_sequence, 0, 1)

            # Normalize patch position and regard padding
            pos[..., 0] = ((pos[..., 0] + 0.5) / (w_tar / 2)) - 1
            pos[..., 1] = ((pos[..., 1] + 0.5) / (h_tar / 2)) - 1
            torch.clamp_(pos, -1, 1)

        return image_sequence, pos.permute(2, 0, 1)

    def drop_below_flow_threshold(self):
        """
        Drops clips from the dataset that do not include sufficient motion according
        to the defined flow threshold.
        Includes a caching mechanism (persisted at `~/.ugc360/config.cache`) to
        avoid recalculation each run.
        """
        if self.flow_threshold <= 0:
            return
        cache_dir = Path.home() / ".ugc360"
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file = cache_dir / "config.cache"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = dict()
        dataset_filenames = [file.stem for file in self.dataset_files]
        cache_key = f"{self.flow_threshold:.10f}_{'_'.join(dataset_filenames)}"
        if cache_key in cache:
            drop_idxs = []
            drop_idxs_string = cache[cache_key]
            if len(drop_idxs_string) > 0:
                drop_idxs = list(map(int, drop_idxs_string.split(",")))
        else:
            print("Looking for clips below flow threshold. This may take some minutes. "
                  "The result is cached at '~/.ugc360/config.cache' for future runs.")
            drop_idxs = []
            for idx, clip in self.df.iterrows():
                clip_path = clip['root'] / str(clip['video_id']) / f"{clip['clip_id']:02d}"
                flow = np.load(clip_path / "flow.npy")
                mask = flow >= self.flow_threshold
                mask = resize(mask, (clip['height'], clip['width']), order=0)
                if np.count_nonzero(mask) == 0:
                    drop_idxs.append(idx)
            cache[cache_key] = ",".join(map(str, drop_idxs))
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
                f.write("\n")
        if len(drop_idxs) > 0:
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
    def patch_rotation_matrix(cls, src_patch_location, tar_patch_location, size_src, size_tar, device=None):
        src_patch_sphere = cls.erp_to_sphere(torch.tensor(src_patch_location), size_src, mode='spherical')
        tar_patch_sphere = cls.erp_to_sphere(torch.tensor(tar_patch_location), size_tar, mode='spherical')
        rotation_matrix_1 = cls.rotation_matrix_z(-tar_patch_sphere[..., 1], device=device)
        rotation_matrix_2 = cls.rotation_matrix_y(-(tar_patch_sphere[..., 0] - src_patch_sphere[..., 0]), device=device)
        rotation_matrix_3 = cls.rotation_matrix_z(src_patch_sphere[..., 1], device=device)
        rotation_matrix = torch.mm(rotation_matrix_3, torch.mm(rotation_matrix_2, rotation_matrix_1))
        return rotation_matrix

    @classmethod
    def sample_target_patch_location(cls, size, patch_size):
        h, w = size
        y = np.random.randint(patch_size[1] // 2, h - patch_size[1] // 2 + 1)
        x = np.random.randint(patch_size[0] // 2, w - patch_size[0] // 2 + 1)
        return np.array([x, y])

    @classmethod
    def sample_src_patch_location(cls, size, mask):
        valid_patch_pos = np.argwhere(mask)
        patch_pos_idx = np.random.randint(0, valid_patch_pos.shape[0])
        patch_pos = valid_patch_pos[patch_pos_idx]

        patch_pos[0] = int(np.round(patch_pos[0] * (size[0] / mask.shape[0])))
        patch_pos[1] = int(np.round(patch_pos[1] * (size[1] / mask.shape[1])))
        return patch_pos[::-1].copy()

    @classmethod
    def get_patch_bounding_box(cls, patch_location, patch_size):
        return (patch_location[0] - patch_size[0] // 2,
                patch_location[1] - patch_size[1] // 2,
                patch_location[0] + patch_size[0] // 2,
                patch_location[1] + patch_size[1] // 2)

    @staticmethod
    def mipmap(images, levels):
        padded_height, padded_width = images.shape[2:4]
        mipmap_pad_width = padded_width // 2
        image_sequence_mipmap = torch.nn.functional.pad(images, (0, mipmap_pad_width, 0, 0, 0, 0),
                                                        mode='constant')
        level_x = 0
        level_y = 0
        level_sub = 1
        for level in range(1, levels + 1):
            last_level_sequence = image_sequence_mipmap[
                                  :,
                                  :,
                                  level_y:level_y + padded_height // level_sub,
                                  level_x:level_x + padded_width // level_sub
                                  ]
            level_x = padded_width if level > 0 else 0
            level_y = level_y + padded_height // level_sub if level > 1 else 0
            level_sub = level_sub * 2
            this_level_sequence = torch.nn.functional.interpolate(last_level_sequence,
                                                                  scale_factor=0.5,
                                                                  mode='bilinear',
                                                                  antialias=True)
            image_sequence_mipmap[:, :, level_y:level_y + padded_height // level_sub,
            level_x:level_x + padded_width // level_sub] = this_level_sequence
        return image_sequence_mipmap
