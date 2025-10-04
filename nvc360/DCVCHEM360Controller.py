import os
from pathlib import Path
from typing import BinaryIO
import torch
import torch.backends.cudnn
import numpy as np
from tqdm import tqdm
from DCVCHEM360.src.models.video_model_posinput import PosInputDMC, PosInputPositions
from DCVCHEM360.src.models.image_model import IntraNoAR
from DCVCHEM360.src.utils.stream_helper import get_state_dict, get_rounded_q
from .CompressionModelController import CompressionModelController, PNGReader, PNGWriter


class DCVCHEM360Controller(CompressionModelController):

    def __init__(self, keyframe_model_path, video_model_path, quality, gop, projection, device='cpu'):
        super().__init__()
        torch.backends.cudnn.enabled = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        np.random.seed(seed=0)

        keyframe_state_dict = get_state_dict(keyframe_model_path)
        self._keyframe_net = IntraNoAR()
        self._keyframe_net.load_state_dict(keyframe_state_dict)
        self._keyframe_net = self._keyframe_net.to(device)
        self._keyframe_net.eval()
        self._keyframe_net.update(force=True)

        video_state_dict = get_state_dict(video_model_path)
        self._video_net = PosInputDMC(posinput_positions=[
            PosInputPositions.HYPERPRIOR_ENCODER,
            PosInputPositions.HYPERPRIOR_DECODER,
            PosInputPositions.ENTROPY_MODEL
        ])
        self._video_net.load_state_dict(video_state_dict)
        self._video_net = self._video_net.to(device)
        self._video_net.eval()
        self._video_net.update(force=True)

        if not 0 <= quality < 4:
            raise ValueError(f"Invalid quality parameter: {quality}. Allowed values are: 0, 1, 2, 3.")

        i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(keyframe_model_path)
        p_frame_y_q_scales, p_frame_mv_y_q_scales = PosInputDMC.get_q_scales_from_ckpt(video_model_path)
        self._q_scale, self._q_index = get_rounded_q(i_frame_q_scales[quality])
        self._mv_y_q_scale, self._mv_y_q_index = get_rounded_q(p_frame_mv_y_q_scales[quality])
        self._y_q_scale, self._y_q_index = get_rounded_q(p_frame_y_q_scales[quality])

        self._projection = projection
        self._gop = gop
        self._device = device


    def pos_encoding(self, width, height):
        if self._projection == 'erp':
            x = torch.arange(0, width, dtype=torch.float32)
            y = torch.arange(0, height, dtype=torch.float32)
            pos = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
            pos[..., 0] = ((pos[..., 0] + 0.5) / (width / 2)) - 1
            pos[..., 1] = ((pos[..., 1] + 0.5) / (height / 2)) - 1
            pos = pos.permute(2, 0, 1).unsqueeze(0).to(self._device)
            return pos
        elif self._projection == 'none':
            pos = torch.zeros((2, height, width), dtype=torch.float32).unsqueeze(0).to(self._device)
            return pos
        else:
            raise NotImplementedError(f"Unknown projection: {self._projection}")


    def encode(self,
               input_path: Path,
               bitstream: BinaryIO,
               width: int,
               height: int,
               bitdepth: int,
               frames: int):
        
        reader = PNGReader(input_path, bitdepth)
        pos_enc = self.pos_encoding(width, height)
        with torch.no_grad():
            for i in tqdm(range(frames), desc="Encoding", unit="frame"):
                x_cur = reader.read(i).to(self._device)
                x_cur = self.pad(x_cur, 64)

                if i % self._gop == 0:
                    encoded = self._keyframe_net.compress(x_cur, self._q_scale)
                    self.write_uints(bitstream, (len(encoded['bit_stream']),))
                    self.write_bytes(bitstream, encoded['bit_stream'])
                    x_ref = self._keyframe_net.decompress(encoded['bit_stream'], height, width, self._q_index / 100)['x_hat']
                    dpb = {
                        "ref_frame": x_ref,
                        "ref_feature": None,
                        "ref_y": None,
                        "ref_mv_y": None,
                    }
                else:
                    encoded = self._video_net.compress(x_cur, dpb, pos_enc, self._mv_y_q_scale, self._y_q_scale)
                    self.write_uints(bitstream, (len(encoded['bit_stream']),))
                    self.write_bytes(bitstream, encoded['bit_stream'])
                    dpb = self._video_net.decompress(dpb, pos_enc, encoded['bit_stream'], height, width,
                                                     self._mv_y_q_index / 100, self._y_q_index / 100)['dpb']


    def decode(self,
               bitstream: BinaryIO,
               output_path: Path,
               width: int,
               height: int,
               bitdepth: int,
               frames: int):
        
        writer = PNGWriter(output_path, bitdepth)
        pos_enc = self.pos_encoding(width, height)
        with torch.no_grad():
            for i in tqdm(range(frames), desc="Decoding", unit="frame"):
                if i % self._gop == 0:
                    num_bytes = self.read_uints(bitstream, 1)[0]
                    bit_stream = self.read_bytes(bitstream, num_bytes)
                    x_ref = self._keyframe_net.decompress(bit_stream, height, width, self._q_index / 100)['x_hat']
                    dpb = {
                        "ref_frame": x_ref,
                        "ref_feature": None,
                        "ref_y": None,
                        "ref_mv_y": None,
                    }
                else:
                    num_bytes = self.read_uints(bitstream, 1)[0]
                    bit_stream = self.read_bytes(bitstream, num_bytes)
                    dpb = self._video_net.decompress(dpb, pos_enc, bit_stream, height, width,
                                                     self._mv_y_q_index / 100, self._y_q_index / 100)['dpb']
                x_ref = dpb['ref_frame'].clamp(0, 1)
                x_rec = self.crop(x_ref, (height, width))
                writer.write(x_rec)
