import random
from typing import Union
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
from DCVCHEM.src.models.video_model import DMC, LowerBound
from DCVCHEM.src.models.image_model import IntraNoAR
from DCVCHEM.src.utils.stream_helper import get_state_dict
from tqdm import tqdm
import datetime
import math
from PIL import Image
import yuvio
import matplotlib.pyplot as plt
import bjontegaard as bd
from DCVCHEM360.datasets import *


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_log(log_writer, loss, mse, bpp, bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, psnr, ws_psnr, data_id, step):
    log_writer.add_scalar(f'loss/{data_id}', loss, step)
    log_writer.add_scalar(f'mse/{data_id}', mse, step)
    log_writer.add_scalar(f'bpp/{data_id}', bpp, step)
    log_writer.add_scalar(f'bpp_mv_y/{data_id}', bpp_mv_y, step)
    log_writer.add_scalar(f'bpp_mv_z/{data_id}', bpp_mv_z, step)
    log_writer.add_scalar(f'bpp_y/{data_id}', bpp_y, step)
    log_writer.add_scalar(f'bpp_z/{data_id}', bpp_z, step)
    log_writer.add_scalar(f'psnr/{data_id}', psnr, step)
    log_writer.add_scalar(f'ws-psnr/{data_id}', ws_psnr, step)


def log_images(log_writer, x, x_hat, step):
    x = x.cpu().detach()
    x_hat = x_hat.cpu().detach()
    images_grid = torch.cat((x, x_hat), dim=0)
    grid = make_grid(images_grid, nrow=x.size(0))
    log_writer.add_image("Input/Output", grid, global_step=step)


def wmse_erp(test, reference, pos):
    # test, ref: [b, c, h, w]
    # pos: [b, h, w, 2 -> (x,y)]
    b, c, h, w = test.shape
    weights = torch.cos(pos[:, 1] * 0.5 * torch.pi)  # [b, h, w]
    weights = weights.unsqueeze(1).expand(-1, c, -1, -1)  # [b, c, h, w]
    wmse_numerator = torch.sum(torch.square(test - reference) * weights, dim=(1, 2, 3))  # [b]
    wmse_denominator = torch.sum(weights, dim=(1, 2, 3))  # [b]
    wmse = wmse_numerator / wmse_denominator
    return wmse


def train_one_epoch(keyframe_model, video_model, optimizer, train_loader, lamdas, epoch, log_writer, val_interval,
                    validation_loader, test_loader, results_anchor):
    keyframe_model.eval()
    video_model.train()
    device = next(video_model.parameters()).device

    train_samples = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    for batch_idx, (image_sequence, pos) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        quality_index = random.randint(0, video_model.anchor_num - 1)
        lamda = lamdas[quality_index]
        q_scale_keyframe = LowerBound.apply(keyframe_model.q_scale[quality_index], 0.5)
        q_scale_motion = LowerBound.apply(video_model.mv_y_q_scale[quality_index], 0.5)
        q_scale_video = LowerBound.apply(video_model.y_q_scale[quality_index], 0.5)

        image_sequence = image_sequence.to(device)
        pos = pos.to(device)

        output = keyframe_model(image_sequence[:, 0], q_scale_keyframe)
        dpb = {
            'ref_frame': output['x_hat'].detach(),
            'ref_feature': None,
            'ref_y': None,
            'ref_mv_y': None
        }
        # Training stage: Multi, 5 frames, cascaded loss
        loss_cascaded = torch.tensor(0.0, device=device)
        n_frames = 5
        avg_bpp = AverageMeter()
        avg_bpp_mv_y = AverageMeter()
        avg_bpp_mv_z = AverageMeter()
        avg_bpp_y = AverageMeter()
        avg_bpp_z = AverageMeter()
        avg_mse = AverageMeter()
        avg_psnr = AverageMeter()
        avg_wspsnr = AverageMeter()
        for frame_idx in range(1, n_frames):
            output = video_model(image_sequence[:, frame_idx],
                                 dpb,
                                 mv_y_q_scale=q_scale_motion,
                                 y_q_scale=q_scale_video)
            dpb = output['dpb']
            wmse = wmse_erp(dpb['ref_frame'], image_sequence[:, frame_idx], pos)
            # loss = lamda * (0.1 * torch.mean(output['mse']) + 0.9 * 2.0 * torch.mean(wmse)) + torch.mean(output['bpp'])
            loss = lamda * torch.mean(output['mse']) + torch.mean(output['bpp'])
            loss_cascaded += loss
            # for k, v in dpb.items():
            #     dpb[k] = v.detach()

            # For logging
            psnr = torch.mean(10 * torch.log10(1.0 / output['mse']))
            ws_psnr = torch.mean(10 * torch.log10(1.0 / wmse))
            avg_bpp.update(torch.mean(output['bpp']).item())
            avg_bpp_mv_y.update(torch.mean(output['bpp_mv_y']).item())
            avg_bpp_mv_z.update(torch.mean(output['bpp_mv_z']).item())
            avg_bpp_y.update(torch.mean(output['bpp_y']).item())
            avg_bpp_z.update(torch.mean(output['bpp_z']).item())
            avg_mse.update(torch.mean(output['mse']).item())
            avg_psnr.update(torch.mean(psnr).item())
            avg_wspsnr.update(torch.mean(ws_psnr).item())

        loss_cascaded = loss_cascaded / (n_frames - 1)
        optimizer.zero_grad()
        loss_cascaded.backward()
        torch.nn.utils.clip_grad_norm_(video_model.parameters(), max_norm=1.0)
        optimizer.step()

        # video_model.update()

        if batch_idx % 100 == 99:
            write_log(
                log_writer,
                loss_cascaded.item(),
                avg_mse.avg,
                avg_bpp.avg,
                avg_bpp_mv_y.avg,
                avg_bpp_mv_z.avg,
                avg_bpp_y.avg,
                avg_bpp_z.avg,
                avg_psnr.avg,
                avg_wspsnr.avg,
                'training',
                epoch * train_samples + batch_idx * batch_size
            )

        if batch_idx % val_interval == val_interval - 1:
            global_step = epoch * train_samples + batch_idx * batch_size
            validate(keyframe_model,
                     video_model,
                     validation_loader,
                     lamdas,
                     log_writer,
                     global_step=global_step)
            test_bd_rate(keyframe_model, video_model, test_loader, results_anchor, log_writer,
                         global_step)


def validate(keyframe_model, video_model, validation_loader, lamdas, log_writer, global_step):
    keyframe_model.eval()
    video_model.eval()
    device = next(video_model.parameters()).device

    avg_loss = AverageMeter()
    avg_bpp = AverageMeter()
    avg_bpp_mv_y = AverageMeter()
    avg_bpp_mv_z = AverageMeter()
    avg_bpp_y = AverageMeter()
    avg_bpp_z = AverageMeter()
    avg_mse = AverageMeter()
    avg_psnr = AverageMeter()
    avg_wspsnr = AverageMeter()
    log_image_batch = random.randint(0, len(validation_loader) - 1)
    with torch.no_grad():
        for batch_idx, (image_sequence, pos) in enumerate(tqdm(validation_loader, desc=f"Validation after step {global_step}")):
            quality_index = random.randint(0, keyframe_net.anchor_num - 1)
            lamda = lamdas[quality_index]
            q_scale_keyframe = LowerBound.apply(keyframe_model.q_scale[quality_index], 0.5)
            q_scale_motion = LowerBound.apply(video_model.mv_y_q_scale[quality_index], 0.5)
            q_scale_video = LowerBound.apply(video_model.y_q_scale[quality_index], 0.5)

            image_sequence = image_sequence.to(device)
            pos = pos.to(device)

            output = keyframe_model(image_sequence[:, 0], q_scale_keyframe)
            dpb = {
                'ref_frame': output['x_hat'],
                'ref_feature': None,
                'ref_y': None,
                'ref_mv_y': None
            }
            # Validation stage: Multi, 5 frames, cascaded loss
            n_frames = 5
            for frame_idx in range(1, n_frames):
                output = video_model(image_sequence[:, frame_idx],
                                     dpb,
                                     mv_y_q_scale=q_scale_motion,
                                     y_q_scale=q_scale_video)
                dpb = output['dpb']
                wmse = wmse_erp(dpb['ref_frame'], image_sequence[:, frame_idx], pos)
                # loss = lamda * (0.1 * torch.mean(output['mse']) + 0.9 * 2.0 * torch.mean(wmse)) + torch.mean(output['bpp'])
                loss = lamda * torch.mean(output['mse']) + torch.mean(output['bpp'])

                psnr = torch.mean(10 * torch.log10(1.0 / output['mse']))
                ws_psnr = torch.mean(10 * torch.log10(1.0 / wmse))

                avg_loss.update(loss.item())
                avg_bpp.update(torch.mean(output['bpp']).item())
                avg_bpp_mv_y.update(torch.mean(output['bpp_mv_y']).item())
                avg_bpp_mv_z.update(torch.mean(output['bpp_mv_z']).item())
                avg_bpp_y.update(torch.mean(output['bpp_y']).item())
                avg_bpp_z.update(torch.mean(output['bpp_z']).item())
                avg_mse.update(torch.mean(output['mse']).item())
                avg_psnr.update(torch.mean(psnr).item())
                avg_wspsnr.update(torch.mean(ws_psnr).item())
            if batch_idx == log_image_batch:
                log_images(log_writer, image_sequence[:, -1], dpb['ref_frame'], global_step)
    write_log(
        log_writer,
        avg_loss.avg,
        avg_mse.avg,
        avg_bpp.avg,
        avg_bpp_mv_y.avg,
        avg_bpp_mv_z.avg,
        avg_bpp_y.avg,
        avg_bpp_z.avg,
        avg_psnr.avg,
        avg_wspsnr.avg,
        'validation',
        global_step
    )
    return avg_loss.avg


def test_rate_distortion(keyframe_model, video_model, test_loader, global_step):
    keyframe_model.eval()
    video_model.eval()

    results = []
    log_image_batch = random.randint(0, len(test_loader) - 1)
    images = []
    with torch.no_grad():
        for batch_idx, (image_sequence, pos) in enumerate(tqdm(test_loader, desc=f"Test after step {global_step}")):

            image_sequence = image_sequence.to(device)
            pos = pos.to(device)

            if batch_idx == log_image_batch:
                images.append(image_sequence[:, -1].detach().cpu())
            result = {
                'sequence': test_loader.dataset.sequence_name(batch_idx),
                'bpp': np.empty(4),
                'psnr': np.empty(4),
                'ws_psnr': np.empty(4)
            }
            for quality_index in range(4):
                q_scale_keyframe = LowerBound.apply(keyframe_model.q_scale[quality_index], 0.5)
                q_scale_motion = LowerBound.apply(video_model.mv_y_q_scale[quality_index], 0.5)
                q_scale_video = LowerBound.apply(video_model.y_q_scale[quality_index], 0.5)

                output = keyframe_model(image_sequence[:, 0], q_scale_keyframe)
                dpb = {
                    'ref_frame': output['x_hat'],
                    'ref_feature': None,
                    'ref_y': None,
                    'ref_mv_y': None
                }
                frame_bpps = []
                frame_psnrs = []
                frame_wspsnrs = []
                for frame_idx in range(1, image_sequence.size(1)):
                    output = video_model(image_sequence[:, frame_idx],
                                         dpb,
                                         mv_y_q_scale=q_scale_motion,
                                         y_q_scale=q_scale_video)
                    dpb = output['dpb']
                    wmse = wmse_erp(dpb['ref_frame'], image_sequence[:, frame_idx], pos)
                    bpp = torch.mean(output['bpp'])
                    psnr = torch.mean(10 * torch.log10(1.0 / output['mse']))
                    ws_psnr = torch.mean(10 * torch.log10(1.0 / wmse))
                    frame_bpps.append(bpp.item())
                    frame_psnrs.append(psnr.item())
                    frame_wspsnrs.append(ws_psnr.item())
                result['bpp'][quality_index] = sum(frame_bpps) / len(frame_bpps)
                result['psnr'][quality_index] = sum(frame_psnrs) / len(frame_psnrs)
                result['ws_psnr'][quality_index] = sum(frame_wspsnrs) / len(frame_wspsnrs)

                if batch_idx == log_image_batch:
                    images.append(dpb['ref_frame'].detach().cpu())
            results.append(result)
    images_grid = torch.cat(images, dim=0)
    grid = make_grid(images_grid, nrow=1)
    log_writer.add_image("Test JVET", grid, global_step=global_step)
    return results


def log_rate_distortion_all(results_anchor, results_test, log_writer, global_step):
    bpp_anchor = np.empty((len(results_anchor), 4))
    psnr_anchor = np.empty((len(results_anchor), 4))
    ws_psnr_anchor = np.empty((len(results_anchor), 4))
    bpp_test = np.empty((len(results_test), 4))
    psnr_test = np.empty((len(results_test), 4))
    ws_psnr_test = np.empty((len(results_test), 4))
    avg_bd_rate_psnr = AverageMeter()
    avg_bd_rate_wspsnr = AverageMeter()
    for i, (result_anchor, result_test) in enumerate(zip(results_anchor, results_test)):
        bd_rate_psnr, bd_rate_wspsnr = log_rate_distortion_for_sequence(
            result_anchor,
            result_test,
            log_writer,
            global_step
        )
        bpp_anchor[i] = result_anchor['bpp']
        psnr_anchor[i] = result_anchor['psnr']
        ws_psnr_anchor[i] = result_anchor['ws_psnr']
        bpp_test[i] = result_test['bpp']
        psnr_test[i] = result_test['psnr']
        ws_psnr_test[i] = result_test['ws_psnr']
        avg_bd_rate_psnr.update(bd_rate_psnr)
        avg_bd_rate_wspsnr.update(bd_rate_wspsnr)

    # Log average rd curve
    bpp_anchor = np.mean(bpp_anchor, axis=0)
    psnr_anchor = np.mean(psnr_anchor, axis=0)
    ws_psnr_anchor = np.mean(ws_psnr_anchor, axis=0)
    bpp_test = np.mean(bpp_test, axis=0)
    psnr_test = np.mean(psnr_test, axis=0)
    ws_psnr_test = np.mean(ws_psnr_test, axis=0)
    result_anchor = {
        'sequence': "Average",
        'bpp': bpp_anchor,
        'psnr': psnr_anchor,
        'ws_psnr': ws_psnr_anchor
    }
    result_test = {
        'sequence': "Average",
        'bpp': bpp_test,
        'psnr': psnr_test,
        'ws_psnr': ws_psnr_test
    }
    _, _ = log_rate_distortion_for_sequence(
        result_anchor,
        result_test,
        log_writer,
        global_step
    )

    avg_bd_rate_psnr = avg_bd_rate_psnr.avg
    avg_bd_rate_wspsnr = avg_bd_rate_wspsnr.avg
    log_writer.add_scalar("bd-rate/psnr", avg_bd_rate_psnr, global_step)
    log_writer.add_scalar("bd-rate/ws-psnr", avg_bd_rate_wspsnr, global_step)


def log_rate_distortion_for_sequence(result_anchor, result_test, log_writer, global_step):
    assert result_anchor["sequence"] == result_test["sequence"]
    sequence = result_anchor["sequence"]

    try:
        output = bd.bd_rate(result_anchor['bpp'], result_anchor['psnr'],
                            result_test['bpp'], result_test['psnr'],
                            method='pchip', min_overlap=0, interpolators=True)
        bd_rate_psnr, interp_anchor_psnr, interp_test_psnr = output
        output = bd.bd_rate(result_anchor['bpp'], result_anchor['ws_psnr'],
                            result_test['bpp'], result_test['ws_psnr'],
                            method='pchip', min_overlap=0, interpolators=True)
        bd_rate_wspsnr, interp_anchor_wspsnr, interp_test_wspsnr = output

        psnrs_anchor = np.linspace(result_anchor['psnr'].min(), result_anchor['psnr'].max(), num=100, endpoint=True)
        wspsnrs_anchor = np.linspace(result_anchor['ws_psnr'].min(), result_anchor['ws_psnr'].max(), num=100, endpoint=True)
        psnrs_test = np.linspace(result_test['psnr'].min(), result_test['psnr'].max(), num=100, endpoint=True)
        wspsnrs_test = np.linspace(result_test['ws_psnr'].min(), result_test['ws_psnr'].max(), num=100, endpoint=True)

        bpps_psnr_anchor = np.power(10, interp_anchor_psnr(psnrs_anchor))
        bpps_wspsnr_anchor = np.power(10, interp_anchor_wspsnr(wspsnrs_anchor))
        bpps_psnr_test = np.power(10, interp_test_psnr(psnrs_test))
        bpps_wspsnr_test = np.power(10, interp_test_wspsnr(wspsnrs_test))

        fig = plt.figure()
        plt.plot(bpps_psnr_anchor, psnrs_anchor, linestyle='-', marker=None, color='tab:blue')
        plt.plot(result_anchor['bpp'], result_anchor['psnr'], linestyle='None', marker='o', color='tab:blue')
        plt.plot(bpps_psnr_test, psnrs_test, linestyle='-', marker=None, color='tab:orange')
        plt.plot(result_test['bpp'], result_test['psnr'], linestyle='None', marker='o', color='tab:orange')

        plt.plot(bpps_wspsnr_anchor, wspsnrs_anchor, linestyle='--', marker=None, color='tab:blue')
        plt.plot(result_anchor['bpp'], result_anchor['ws_psnr'], linestyle='None', marker='o', color='tab:blue')
        plt.plot(bpps_wspsnr_test, wspsnrs_test, linestyle='--', marker=None, color='tab:orange')
        plt.plot(result_test['bpp'], result_test['ws_psnr'], linestyle='None', marker='o', color='tab:orange')

        plt.grid()
        plt.xlabel("bpp in bit")
        plt.ylabel("(WS-)PSNR in dB")
        plt.title(f"{sequence} | {bd_rate_psnr:.2f}% (PSNR) | {bd_rate_wspsnr:.2f}% (WS-PSNR)")
        log_writer.add_figure(f"RD-Plot {sequence}", fig, global_step)
        plt.close(fig)
    except:
        return float('nan'), float('nan')

    return bd_rate_psnr, bd_rate_wspsnr


def test_bd_rate(keyframe_model, video_model, test_loader, results_anchor, log_writer, global_step):
    results_test = test_rate_distortion(keyframe_model, video_model, test_loader, global_step)
    log_rate_distortion_all(results_anchor, results_test, log_writer, global_step)


def setup_dataloaders(dataset_name, batch_size, patch_size=(256, 256)):
    if dataset_name == "vimeo90k":
        train_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop(patch_size)]  # transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()
        )
        test_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(patch_size)]
        )
        train_set = Vimeo90kDataset("/home/fa94ciqu/Resources/vimeo_septuplet", sequence_length=5,
                                    transform=train_transforms, split="train", tuplet=7)
        valid_set = Vimeo90kDataset("/home/fa94ciqu/Resources/vimeo_septuplet", sequence_length=5,
                                    transform=test_transforms, split="valid", tuplet=7)
    elif dataset_name == "dataset360":
        # FIXME: Validation set should use test_transforms
        data_set = Dataset360("/home/fa94ciqu/Resources/dataset360/clips",
                              sequence_length=5,
                              flow_threshold=0.5,
                              mipmap_levels=8,
                              max_anisotropy=8)
        train_set, valid_set = torch.utils.data.random_split(data_set, [0.9, 0.1])
    elif dataset_name == "vimeo90k360":
        # FIXME: Validation set should use test_transforms
        train_set = Vimeo90K360("/home/fa94ciqu/Resources/vimeo_septuplet",
                                sequence_length=5,
                                split="train",
                                tuplet=7,
                                erp_size_range=(512, 2048))
        valid_set = Vimeo90K360("/home/fa94ciqu/Resources/vimeo_septuplet",
                                sequence_length=5,
                                split="valid",
                                tuplet=7,
                                erp_size_range=(512, 2048))
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, valid_loader


def setup_keyframe_model(model_path, device):
    keyframe_net = IntraNoAR()
    keyframe_state_dict = get_state_dict(model_path)
    keyframe_net.load_state_dict(keyframe_state_dict)
    keyframe_net.to(device)
    return keyframe_net


def setup_video_model(model_path, device):
    video_net = DMC()
    video_state_dict = get_state_dict(model_path)
    video_net.load_state_dict(video_state_dict)
    video_net.to(device)
    return video_net


def test_rd_anchor(test_loader):
    anchor_net = IntraNoAR()
    keyframe_model_path = Path("/home/fa94ciqu/Development/DCVC/DCVCHEM/checkpoints/acmmm2022_image_psnr.pth.tar")
    keyframe_state_dict = get_state_dict(keyframe_model_path)
    anchor_net.load_state_dict(keyframe_state_dict)
    anchor_net = anchor_net.to(device)
    video_anchor_net = DMC()
    video_state_dict = get_state_dict("/home/fa94ciqu/Development/DCVC/DCVCHEM/checkpoints/acmmm2022_video_psnr.pth.tar")
    video_anchor_net.load_state_dict(video_state_dict)
    video_anchor_net.to(device)
    results_anchor = test_rate_distortion(anchor_net, video_anchor_net, test_loader, 0)
    return results_anchor


if __name__ == '__main__':
    # Setup environment
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    torch.manual_seed(1234)
    np.random.seed(seed=1234)
    random.seed(1234)

    # Parameters
    batch_size = 4
    patch_size = (256, 256)
    pretrained = True
    lr = 1e-5 if pretrained else 1e-4

    # Setup tensorboard log writer
    run_path = 'runs/vimeo90kfine_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    log_writer = SummaryWriter(run_path)
    log_writer.add_custom_scalars({
        "DCVC-HEM (DMC)": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "mse": ["Multiline", ["mse/train", "mse/validation"]],
            "bpp": ["Multiline", ["bpp/train", "bpp/validation"]],
            "psnr": ["Multiline", ["psnr/train", "psnr/validation"]],
            "ws-psnr": ["Multiline", ["ws-psnr/train", "ws-psnr/validation"]],
            "bd-rate": ["Multiline", ["bd-rate/psnr", "bd-rate/ws-psnr"]],
            "bpp_mv_y": ["Multiline", ["bpp_mv_y/train", "bpp_mv_y/validation"]],
            "bpp_mv_z": ["Multiline", ["bpp_mv_z/train", "bpp_mv_z/validation"]],
            "bpp_y": ["Multiline", ["bpp_y/train", "bpp_y/validation"]],
            "bpp_z": ["Multiline", ["bpp_z/train", "bpp_z/validation"]]
        },
    })

    # Setup training, validation and test data loaders
    train_loader, valid_loader = setup_dataloaders("vimeo90k", batch_size, patch_size)
    jvet_testset = JVET360Dataset("/home/fa94ciqu/Resources/jvet/360", 8, (1024, 2048))
    test_loader = DataLoader(jvet_testset, shuffle=False, batch_size=1, num_workers=4)

    # Setup models
    keyframe_net = setup_keyframe_model(
        "/home/fa94ciqu/Development/DCVC/DCVCHEM/checkpoints/acmmm2022_image_psnr.pth.tar",
        device=device
    )
    video_net = setup_video_model(
        "/home/fa94ciqu/Development/DCVC/DCVCHEM/checkpoints/acmmm2022_video_psnr.pth.tar",
        device=device
    )

    # Prepare lamda values for different quality indices (according to paper)
    lamdas = [85, 170, 380, 840]
    assert(len(lamdas) == keyframe_net.anchor_num)

    # Prepare optimizer and learning rate scheduler
    all_wo_optic_flow_parameters = [parameter for name, parameter in video_net.named_parameters() if "optic_flow" not in name]
    optimizer = optim.SGD([
        {'params': all_wo_optic_flow_parameters},
    ], lr=lr, momentum=0.9)
    print(lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    # vimeo90kfine_2024-03-18_16:40: Adam
    # vimeo90kfine_2024-03-18_19:06: SGD with momentum
    # vimeo90kfine_2024-03-19_09:47: SGD
    # vimeo90kfine_2024-03-19_15:41: SGD lr=5e-6
    # vimeo90kfine_2024-03-20_08:55: SGD lr=1e-5 without flip augmentation and val_interval 1000 (STABLE)
    # vimeo90kfine_2024-03-20_22:18: SGD lr=1e-5 with momentum=0.9 without flip augmentation and val_interval 1000
    # vimeo90kfine_2024-03-19_00:24: Adam with betas (0, 0.999)
    # : Adam with betas (0.9, 0)
    # : Adam with betas (0, 0)

    # Prepare training
    best_loss = float('inf')
    if pretrained:
        best_loss = validate(keyframe_net, video_net, valid_loader, lamdas, log_writer, 0)
        results_anchor = test_rd_anchor(test_loader)

    # Run training loop
    for epoch in range(10000):
        # configure_training_stage(keyframe_net, optimizer, epoch, pretrained)
        train_one_epoch(keyframe_net, video_net, optimizer, train_loader, lamdas, epoch, log_writer,
                        1000, valid_loader, test_loader, results_anchor)
        loss = validate(keyframe_net,
                        video_net,
                        valid_loader,
                        lamdas,
                        log_writer,
                        global_step=(epoch + 1) * len(train_loader.dataset))
        scheduler.step(loss)

        test_bd_rate(keyframe_net, video_net, test_loader, results_anchor, log_writer, (epoch + 1) * len(train_loader.dataset))
        # log_spatial_quant(keyframe_net.q_spatial, log_writer, (epoch + 1) * len(train_loader.dataset))

        # Save training checkpoint
        checkpoint_path = Path(run_path) / "checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': video_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)

        # Save best model checkpoint if validation loss improved
        best_checkpoint_path = Path(run_path) / "checkpoint_best.pth"
        if loss < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': video_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, best_checkpoint_path)


    log_writer.flush()
