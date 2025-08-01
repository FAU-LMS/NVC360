# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time

import torch
from torch import nn
from enum import Enum

from .common_model import CompressionModel
from .video_net import ME_Spynet, flow_warp, bilineardownsacling, LowerBound
from ..layers.layers import conv3x3, subpel_conv1x1
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, get_rounded_q, get_state_dict
from .pos_encoding import PosEncoder, PosDownsampler
from .modules import (FeatureExtractor, MultiScaleContextFusion, ReconGeneration, ContextualEncoder, ContextualDecoder,
                      PosContextualEncoder, PosContextualDecoder, PosReconGeneration,
                      get_enc_dec_models, get_hyper_enc_dec_models)


class PosInputPositions(Enum):
    CONTEXTUAL_ENCODER = "contextual_encoder"
    CONTEXTUAL_DECODER = "contextual_decoder"
    FRAME_GENERATOR = "frame_generator"
    CONTEXT_GENERATOR = "context_generator"
    HYPERPRIOR_ENCODER = "hyperprior_encoder"
    HYPERPRIOR_DECODER = "hyperprior_decoder"
    ENTROPY_MODEL = "entropy_model"


class PosInputDMC(CompressionModel):
    def __init__(self,
                 anchor_num=4,
                 posinput_positions=(PosInputPositions.CONTEXTUAL_ENCODER,
                                     PosInputPositions.CONTEXTUAL_DECODER,
                                     PosInputPositions.FRAME_GENERATOR,
                                     PosInputPositions.CONTEXT_GENERATOR,
                                     PosInputPositions.HYPERPRIOR_ENCODER,
                                     PosInputPositions.HYPERPRIOR_DECODER,
                                     PosInputPositions.ENTROPY_MODEL)
                 ):
        super().__init__(y_distribution='laplace', z_channel=64, mv_z_channel=64)
        self.DMC_version = '1.19'

        channel_mv = 64
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        self.optic_flow = ME_Spynet()

        self.pos_encoder = PosEncoder()
        channel_pos = self.pos_encoder.channels
        self.pos_downsampler = PosDownsampler()
        self.posinput_positions = posinput_positions

        self.mv_encoder, self.mv_decoder = get_enc_dec_models(2, 2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N)

        self.mv_y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1)
        )

        self.mv_y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_mv * 4, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 2, 3, padding=1)
        )

        if PosInputPositions.CONTEXT_GENERATOR in self.posinput_positions:
            feature_adaptor_channel_pos = channel_pos
        else:
            feature_adaptor_channel_pos = 0
        self.feature_adaptor_I = nn.Conv2d(3 + feature_adaptor_channel_pos, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N + feature_adaptor_channel_pos, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        if PosInputPositions.CONTEXTUAL_ENCODER in self.posinput_positions:
            self.contextual_encoder = PosContextualEncoder(channel_N=channel_N, channel_M=channel_M, channel_pos=channel_pos)
        else:
            self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)

        if PosInputPositions.HYPERPRIOR_ENCODER in self.posinput_positions:
            hyperprior_encoder_channel_pos = channel_pos
        else:
            hyperprior_encoder_channel_pos = 0
        self.contextual_hyper_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M + hyperprior_encoder_channel_pos, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        if PosInputPositions.HYPERPRIOR_DECODER in self.posinput_positions:
            hyperprior_decoder_channel_pos = channel_pos
        else:
            hyperprior_decoder_channel_pos = 0
        self.contextual_hyper_prior_decoder = nn.Sequential(
            conv3x3(channel_N + hyperprior_decoder_channel_pos, channel_M),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M, channel_M, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M, channel_M * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M * 3 // 2, channel_M * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M * 3 // 2, channel_M * 2),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_N, channel_M * 3 // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1),
        )

        if PosInputPositions.ENTROPY_MODEL in self.posinput_positions:
            y_prior_fusion_channel_pos = channel_pos
        else:
            y_prior_fusion_channel_pos = 0
        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_M * 5 + y_prior_fusion_channel_pos, channel_M * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 2, 3, padding=1)
        )

        if PosInputPositions.CONTEXTUAL_DECODER in self.posinput_positions:
            self.contextual_decoder = PosContextualDecoder(channel_N=channel_N, channel_M=channel_M, channel_pos=channel_pos)
        else:
            self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)
        if PosInputPositions.FRAME_GENERATOR in self.posinput_positions:
            self.recon_generation_net = PosReconGeneration(channel_pos=channel_pos)
        else:
            self.recon_generation_net = ReconGeneration()

        self.mv_y_q_basic = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_basic = nn.Parameter(torch.ones((1, channel_M, 1, 1)))
        self.y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.anchor_num = int(anchor_num)

        self._initialize_weights()

    def multi_scale_feature_extractor(self, dpb, pos_enc):
        if PosInputPositions.CONTEXT_GENERATOR in self.posinput_positions:
            pos_enc = pos_enc
        else:
            b, c, h, w = pos_enc.shape
            pos_enc = torch.empty((b, 0, h, w), device=pos_enc.device)  # Such that pos_enc concatenation has no effect
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(torch.cat([dpb["ref_frame"], pos_enc], dim=1))
        else:
            feature = self.feature_adaptor_P(torch.cat([dpb["ref_feature"], pos_enc], dim=1))
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, pos_enc, mv):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb, pos_enc)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scales = ckpt["y_q_scale"]
        mv_y_q_scales = ckpt["mv_y_q_scale"]
        return y_q_scales.reshape(-1), mv_y_q_scales.reshape(-1)

    def get_curr_mv_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.mv_y_q_basic, 0.5)
        return q_basic * q_scale

    def get_curr_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.y_q_basic, 0.5)
        return q_basic * q_scale

    def compress(self, x, dpb, pos, mv_y_q_scale, y_q_scale):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
        curr_y_q = self.get_curr_y_q(y_q_scale)
        pos_enc = self.pos_encoder(pos)
        pos_enc_ds = self.pos_downsampler(pos_enc)

        est_mv = self.optic_flow(x, dpb["ref_frame"])
        mv_y = self.mv_encoder(est_mv)
        mv_y = mv_y / curr_mv_y_q
        mv_z = self.mv_hyper_prior_encoder(mv_y)
        mv_z_hat = torch.round(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_q_w_0, mv_y_q_w_1, mv_scales_w_0, mv_scales_w_1, mv_y_hat = self.compress_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, _ = self.motion_compensation(dpb, pos_enc, mv_hat)

        if PosInputPositions.CONTEXTUAL_ENCODER in self.posinput_positions:
            y = self.contextual_encoder(x, pos_enc, context1, context2, context3)
        else:
            y = self.contextual_encoder(x, context1, context2, context3)
        y = y / curr_y_q
        if PosInputPositions.HYPERPRIOR_ENCODER in self.posinput_positions:
            z = self.contextual_hyper_prior_encoder(torch.cat([y, pos_enc_ds['ds16']], dim=1))
        else:
            z = self.contextual_hyper_prior_encoder(y)
        z_hat = torch.round(z)
        if PosInputPositions.HYPERPRIOR_DECODER in self.posinput_positions:
            hierarchical_params = self.contextual_hyper_prior_decoder(torch.cat([z_hat, pos_enc_ds['ds64']], dim=1))
        else:
            hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        if PosInputPositions.ENTROPY_MODEL in self.posinput_positions:
            params = torch.cat((temporal_params, hierarchical_params, ref_y, pos_enc_ds['ds16']), dim=1)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)

        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat = self.compress_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        if PosInputPositions.CONTEXTUAL_DECODER in self.posinput_positions:
            recon_image_feature = self.contextual_decoder(y_hat, pos_enc_ds['ds16'], context2, context3)
        else:
            recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        if PosInputPositions.FRAME_GENERATOR in self.posinput_positions:
            feature, x_hat = self.recon_generation_net(recon_image_feature, context1, pos_enc)
        else:
            feature, x_hat = self.recon_generation_net(recon_image_feature, context1)

        self.entropy_coder.reset_encoder()
        _ = self.bit_estimator_z_mv.encode(mv_z_hat)
        _ = self.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        _ = self.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        _ = self.bit_estimator_z.encode(z_hat)
        _ = self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        _ = self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        bit_stream = self.entropy_coder.flush_encoder()

        result = {
            "dbp": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, dpb, pos, string, height, width,
                   mv_y_q_scale, y_q_scale):
        curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
        curr_y_q = self.get_curr_y_q(y_q_scale)
        pos_enc = self.pos_encoder(pos)
        pos_enc_ds = self.pos_downsampler(pos_enc)

        self.entropy_coder.set_stream(string)
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(mv_z_size)
        mv_z_hat = mv_z_hat.to(device)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            _, C, H, W = mv_params.size()
            ref_mv_y = torch.zeros((1, C // 2, H, W), device=mv_params.device)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_hat = self.decompress_dual_prior(mv_means, mv_scales, mv_q_step,
                                              self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, _ = self.motion_compensation(dpb, pos_enc, mv_hat)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bit_estimator_z.decode_stream(z_size)
        z_hat = z_hat.to(device)
        if PosInputPositions.HYPERPRIOR_DECODER in self.posinput_positions:
            hierarchical_params = self.contextual_hyper_prior_decoder(torch.cat([z_hat, pos_enc_ds['ds64']], dim=1))
        else:
            hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if ref_y is None:
            _, C, H, W = temporal_params.size()
            ref_y = torch.zeros((1, C // 2, H, W), device=temporal_params.device)
        if PosInputPositions.ENTROPY_MODEL in self.posinput_positions:
            params = torch.cat((temporal_params, hierarchical_params, ref_y, pos_enc_ds['ds16']), dim=1)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_hat = self.decompress_dual_prior(means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        if PosInputPositions.CONTEXTUAL_DECODER in self.posinput_positions:
            recon_image_feature = self.contextual_decoder(y_hat, pos_enc_ds['ds16'], context2, context3)
        else:
            recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        if PosInputPositions.FRAME_GENERATOR in self.posinput_positions:
            feature, recon_image = self.recon_generation_net(recon_image_feature, context1, pos_enc)
        else:
            feature, recon_image = self.recon_generation_net(recon_image_feature, context1)
        recon_image = recon_image.clamp(0, 1)

        return {
            "dpb": {
                "ref_frame": recon_image,
                "ref_feature": feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
        }

    def encode_decode(self, x, dpb, pos, output_path=None,
                      pic_width=None, pic_height=None,
                      mv_y_q_scale=None, y_q_scale=None):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path is not None:
            mv_y_q_scale, mv_y_q_index = get_rounded_q(mv_y_q_scale)
            y_q_scale, y_q_index = get_rounded_q(y_q_scale)

            encoded = self.compress(x, dpb, pos,
                                    mv_y_q_scale, y_q_scale)
            encode_p(encoded['bit_stream'], mv_y_q_index, y_q_index, output_path)
            bits = filesize(output_path) * 8
            mv_y_q_index, y_q_index, string = decode_p(output_path)

            start = time.time()
            decoded = self.decompress(dpb, pos, string,
                                      pic_height, pic_width,
                                      mv_y_q_index / 100, y_q_index / 100)
            decoding_time = time.time() - start
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "decoding_time": decoding_time,
            }
            return result

        encoded = self.forward_one_frame(x, dpb, pos,
                                         mv_y_q_scale=mv_y_q_scale, y_q_scale=y_q_scale)
        result = {
            "dpb": encoded['dpb'],
            "bit_y": encoded['bit_y'].item(),
            "bit_z": encoded['bit_z'].item(),
            "bit_mv_y": encoded['bit_mv_y'].item(),
            "bit_mv_z": encoded['bit_mv_z'].item(),
            "bit": encoded['bit'].item(),
            "decoding_time": 0,
        }
        return result

    def forward_one_frame(self, x, dpb, pos, mv_y_q_scale=None, y_q_scale=None):
        ref_frame = dpb["ref_frame"]
        curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
        curr_y_q = self.get_curr_y_q(y_q_scale)
        pos_enc = self.pos_encoder(pos)
        pos_enc_ds = self.pos_downsampler(pos_enc)

        est_mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_y = mv_y / curr_mv_y_q
        mv_z = self.mv_hyper_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, pos_enc, mv_hat)

        if PosInputPositions.CONTEXTUAL_ENCODER in self.posinput_positions:
            y = self.contextual_encoder(x, pos_enc, context1, context2, context3)
        else:
            y = self.contextual_encoder(x, context1, context2, context3)
        y = y / curr_y_q
        if PosInputPositions.HYPERPRIOR_ENCODER in self.posinput_positions:
            z = self.contextual_hyper_prior_encoder(torch.cat([y, pos_enc_ds['ds16']], dim=1))
        else:
            z = self.contextual_hyper_prior_encoder(y)
        z_hat = self.quant(z)
        if PosInputPositions.HYPERPRIOR_DECODER in self.posinput_positions:
            hierarchical_params = self.contextual_hyper_prior_decoder(torch.cat([z_hat, pos_enc_ds['ds64']], dim=1))
        else:
            hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)

        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        if PosInputPositions.ENTROPY_MODEL in self.posinput_positions:
            params = torch.cat((temporal_params, hierarchical_params, ref_y, pos_enc_ds['ds16']), dim=1)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        if PosInputPositions.CONTEXTUAL_DECODER in self.posinput_positions:
            recon_image_feature = self.contextual_decoder(y_hat, pos_enc_ds['ds16'], context2, context3)
        else:
            recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        if PosInputPositions.FRAME_GENERATOR in self.posinput_positions:
            feature, recon_image = self.recon_generation_net(recon_image_feature, context1, pos_enc)
        else:
            feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        B, _, H, W = x.size()
        pixel_num = H * W
        mse = self.mse(x, recon_image)
        ssim = self.ssim(x, recon_image)
        me_mse = self.mse(x, warp_frame)
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num
        me_mse = torch.sum(me_mse, dim=(1, 2, 3)) / pixel_num

        if self.training:
            y_for_bit = self.add_noise(y_res)
            mv_y_for_bit = self.add_noise(mv_y_res)
            z_for_bit = self.add_noise(z)
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            y_for_bit = y_q
            mv_y_for_bit = mv_y_q
            z_for_bit = z_hat
            mv_z_for_bit = mv_z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "me_mse": me_mse,
                "mse": mse,
                "ssim": ssim,
                "dpb": {
                    "ref_frame": recon_image,
                    "ref_feature": feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                "bitmap_y": torch.sum(bits_y, dim=1),
                "bitmap_z": torch.sum(bits_z, dim=1),
                "bitmap_mv_y": torch.sum(bits_mv_y, dim=1),
                "bitmap_mv_z": torch.sum(bits_mv_z, dim=1)
                }

    def forward(self, x, dpb, pos, mv_y_q_scale=None, y_q_scale=None):
        return self.forward_one_frame(x, dpb, pos,
                                      mv_y_q_scale=mv_y_q_scale, y_q_scale=y_q_scale)
