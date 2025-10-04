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
from DCVCHEM.src.utils.stream_helper import get_state_dict, get_rounded_q
from tqdm import tqdm
import datetime
import math
from PIL import Image
import yuvio
import matplotlib.pyplot as plt
import bjontegaard as bd
from dataset import Dataset360, Vimeo90kDataset, JVET360Dataset


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


def write_log(log_writer, loss, mse, bpp, psnr, ws_psnr, data_id, step):
    log_writer.add_scalar(f'loss/{data_id}', loss, step)
    log_writer.add_scalar(f'mse/{data_id}', mse, step)
    log_writer.add_scalar(f'bpp/{data_id}', bpp, step)
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
    weights = torch.cos(pos[..., 1] * 0.5 * torch.pi)  # [b, h, w]
    weights = weights.unsqueeze(1).expand(-1, c, -1, -1)  # [b, c, h, w]
    wmse_numerator = torch.sum(torch.square(test - reference) * weights, dim=(1, 2, 3))  # [b]
    wmse_denominator = torch.sum(weights, dim=(1, 2, 3))  # [b]
    wmse = wmse_numerator / wmse_denominator
    return wmse


def train_one_epoch(model, optimizer, train_loader, lamdas, epoch, log_writer):
    model.train()
    device = next(model.parameters()).device

    train_samples = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    for batch_idx, (image, pos) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        quality_index = random.randint(0, keyframe_net.anchor_num - 1)
        lamda = lamdas[quality_index]
        q_scale = LowerBound.apply(model.q_scale[quality_index], 0.5)

        image = image[:, 0].to(device)
        pos = pos.to(device)

        optimizer.zero_grad()
        output = model(image, q_scale)
        wmse = wmse_erp(output['x_hat'], image, pos)
        # loss = 2.5 * lamda * (0.1 * torch.mean(output['mse']) + 2.0 * 0.9 * torch.mean(wmse)) + torch.mean(output['bpp'])
        loss = 2.5 * lamda * torch.mean(output['mse']) + torch.mean(output['bpp'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if batch_idx % 100 == 99:
            wmse = wmse_erp(output['x_hat'], image, pos)
            psnr = 10 * torch.log10(1.0 / output['mse'])
            ws_psnr = 10 * torch.log10(1.0 / wmse)
            write_log(
                log_writer,
                loss.item(),
                torch.mean(output['mse']),
                torch.mean(output['bpp']),
                torch.mean(psnr),
                torch.mean(ws_psnr),
                'training',
                epoch * train_samples + batch_idx * batch_size
            )


def validate(model, validation_loader, lamdas, log_writer, global_step):
    model.eval()
    device = next(model.parameters()).device

    avg_loss = AverageMeter()
    avg_mse = AverageMeter()
    avg_bpp = AverageMeter()
    avg_psnr = AverageMeter()
    avg_ws_psnr = AverageMeter()
    log_image_batch = random.randint(0, len(validation_loader) - 1)
    with torch.no_grad():
        for batch_idx, (image, pos) in enumerate(tqdm(validation_loader, desc=f"Validation after step {global_step}")):
            quality_index = random.randint(0, keyframe_net.anchor_num - 1)
            lamda = lamdas[quality_index]
            q_scale = LowerBound.apply(model.q_scale[quality_index], 0.5)

            image = image[:, 0].to(device)
            pos = pos.to(device)

            output = model(image, q_scale)
            wmse = wmse_erp(output['x_hat'], image, pos)
            # loss = 2.5 * lamda * (0.1 * torch.mean(output['mse']) + 2.0 * 0.9 * torch.mean(wmse)) + torch.mean(output['bpp'])
            loss = 2.5 * lamda * torch.mean(output['mse']) + torch.mean(output['bpp'])
            mse = torch.mean(output['mse'])
            bpp = torch.mean(output['bpp'])
            psnr = torch.mean(10 * torch.log10(1.0 / output['mse']))
            ws_psnr = torch.mean(10 * torch.log10(1.0 / wmse))
            avg_loss.update(loss.item())
            avg_mse.update(mse.item())
            avg_bpp.update(bpp.item())
            avg_psnr.update(psnr.item())
            avg_ws_psnr.update(ws_psnr.item())
            if batch_idx == log_image_batch:
                log_images(log_writer, image, output['x_hat'], global_step)
    write_log(
        log_writer,
        avg_loss.avg,
        avg_mse.avg,
        avg_bpp.avg,
        avg_psnr.avg,
        avg_ws_psnr.avg,
        'validation',
        global_step
    )
    return avg_loss.avg


def test_rate_distortion(model, test_loader, global_step):
    model.eval()

    results = []
    log_image_batch = random.randint(0, len(test_loader) - 1)
    images = []
    with torch.no_grad():
        for batch_idx, (image, pos) in enumerate(tqdm(test_loader, desc=f"Test after step {global_step}")):
            image = image[:, 0].to(device)
            pos = pos.to(device)
            if batch_idx == log_image_batch:
                images.append(image.detach().cpu())
            result = {
                'bpp': [],
                'psnr': [],
                'ws_psnr': []
            }
            for quality_index in range(4):
                q_scale = LowerBound.apply(model.q_scale[quality_index], 0.5)

                output = model(image, q_scale)
                wmse = wmse_erp(output['x_hat'], image, pos)
                bpp = torch.mean(output['bpp'])
                psnr = torch.mean(10 * torch.log10(1.0 / output['mse']))
                ws_psnr = torch.mean(10 * torch.log10(1.0 / wmse))
                result['bpp'].append(bpp.item())
                result['psnr'].append(psnr.item())
                result['ws_psnr'].append(ws_psnr.item())
                if batch_idx == log_image_batch:
                    images.append(output['x_hat'].detach().cpu())
            results.append(result)
    images_grid = torch.cat(images, dim=0)
    grid = make_grid(images_grid, nrow=1)
    log_writer.add_image("Test JVET", grid, global_step=global_step)
    return results


def log_rd_curve(results_anchor, results_test, log_writer, global_step):
    bpp_anchor = np.empty((len(results_anchor), 4))
    psnr_anchor = np.empty((len(results_anchor), 4))
    ws_psnr_anchor = np.empty((len(results_anchor), 4))
    bpp_test = np.empty((len(results_test), 4))
    psnr_test = np.empty((len(results_test), 4))
    ws_psnr_test = np.empty((len(results_test), 4))
    for i, (result_anchor, result_test) in enumerate(zip(results_anchor, results_test)):
        bpp_anchor[i] = result_anchor['bpp']
        psnr_anchor[i] = result_anchor['psnr']
        ws_psnr_anchor[i] = result_anchor['ws_psnr']
        bpp_test[i] = result_test['bpp']
        psnr_test[i] = result_test['psnr']
        ws_psnr_test[i] = result_test['ws_psnr']
    bpp_anchor = np.mean(bpp_anchor, axis=0)
    psnr_anchor = np.mean(psnr_anchor, axis=0)
    ws_psnr_anchor = np.mean(ws_psnr_anchor, axis=0)
    bpp_test = np.mean(bpp_test, axis=0)
    psnr_test = np.mean(psnr_test, axis=0)
    ws_psnr_test = np.mean(ws_psnr_test, axis=0)

    fig = plt.figure()
    plt.plot(bpp_anchor, psnr_anchor, linestyle='-', marker='o', color='tab:blue')
    plt.plot(bpp_test, psnr_test, linestyle='-', marker='o', color='tab:orange')
    plt.plot(bpp_anchor, ws_psnr_anchor, linestyle='--', marker='o', color='tab:blue')
    plt.plot(bpp_test, ws_psnr_test, linestyle='--', marker='o', color='tab:orange')
    plt.grid()
    plt.xlabel("bpp in bit")
    plt.ylabel("(WS-)PSNR in dB")
    log_writer.add_figure("RD-Plot", fig, global_step)
    plt.close(fig)


def test_bd_rate(model, test_loader, results_anchor, log_writer, global_step):
    results_test = test_rate_distortion(model, test_loader, global_step)
    avg_bd_rate_psnr = AverageMeter()
    avg_bd_rate_wspsnr = AverageMeter()
    for result_anchor, result_test in zip(results_anchor, results_test):
        bd_rate_psnr = bd.bd_rate(result_anchor['bpp'], result_anchor['psnr'],
                                  result_test['bpp'], result_test['psnr'],
                                  method='pchip', min_overlap=0)
        bd_rate_wspsnr = bd.bd_rate(result_anchor['bpp'], result_anchor['ws_psnr'],
                                    result_test['bpp'], result_test['ws_psnr'],
                                    method='pchip', min_overlap=0)
        avg_bd_rate_psnr.update(bd_rate_psnr)
        avg_bd_rate_wspsnr.update(bd_rate_wspsnr)
    avg_bd_rate_psnr = avg_bd_rate_psnr.avg
    avg_bd_rate_wspsnr = avg_bd_rate_wspsnr.avg
    log_writer.add_scalar("bd-rate/psnr", avg_bd_rate_psnr, global_step)
    log_writer.add_scalar("bd-rate/ws-psnr", avg_bd_rate_wspsnr, global_step)
    log_rd_curve(results_anchor, results_test, log_writer, global_step)


def setup_dataloaders(dataset_name, batch_size, patch_size=(256, 256)):
    if dataset_name == "vimeo90k":
        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomCrop(patch_size)]
        )
        test_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(patch_size)]
        )
        train_set = Vimeo90kDataset("/home/fa94ciqu/Resources/vimeo_septuplet", sequence_length=1,
                                    transform=train_transforms, split="train", tuplet=7)
        valid_set = Vimeo90kDataset("/home/fa94ciqu/Resources/vimeo_septuplet", sequence_length=1,
                                    transform=test_transforms, split="valid", tuplet=7)
    elif dataset_name == "dataset360":
        # FIXME: Validation set should use test_transforms
        data_set = Dataset360("/home/fa94ciqu/Resources/dataset360/clips", sequence_length=1)
        train_set, valid_set = torch.utils.data.random_split(data_set, [0.9, 0.1])
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, valid_loader


def setup_model(pretrained=False):
    keyframe_net = IntraNoAR()
    if pretrained:
        keyframe_model_path = Path("/home/fa94ciqu/Development/DCVC/DCVCHEM/checkpoints/acmmm2022_image_psnr.pth.tar")
        keyframe_state_dict = get_state_dict(keyframe_model_path)
        keyframe_net.load_state_dict(keyframe_state_dict)
    keyframe_net = keyframe_net.to(device)
    # Probably, the next step may not be done for training, because an actual entropy coder is included otherwise
    # keyframe_net.update(force=True)
    return keyframe_net


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
    run_path = 'runs/keyframe_model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    log_writer = SummaryWriter(run_path)
    log_writer.add_custom_scalars({
        "Pretrained DCVC-HEM (IntraNoAR)": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "mse": ["Multiline", ["mse/train", "mse/validation"]],
            "bpp": ["Multiline", ["bpp/train", "bpp/validation"]],
            "psnr": ["Multiline", ["psnr/train", "psnr/validation"]],
            "ws-psnr": ["Multiline", ["ws-psnr/train", "ws-psnr/validation"]],
            "bd-rate": ["Multiline", ["bd-rate/psnr", "bd-rate/ws-psnr"]]
        },
    })

    # Setup training, validation and test data loaders
    train_loader, valid_loader = setup_dataloaders("vimeo90k", batch_size, patch_size)
    jvet_testset = JVET360Dataset("/CLUSTERHOMES/LMS/sequences/jvet/360", 1, (1024, 2048))
    test_loader = DataLoader(jvet_testset, shuffle=False, batch_size=1, num_workers=4)

    # Setup model
    keyframe_net = setup_model(pretrained=pretrained)

    # Prepare lamda values for different quality indices (according to paper)
    lamdas = [85, 170, 380, 840]
    assert(len(lamdas) == keyframe_net.anchor_num)

    # Prepare optimizer and learning rate scheduler
    optimizer = optim.AdamW(keyframe_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)

    # Prepare training
    best_loss = float('inf')
    if pretrained:
        best_loss = validate(keyframe_net, valid_loader, lamdas, log_writer, 0)
        results_anchor = test_rate_distortion(keyframe_net, test_loader, 0)

    # Run training loop
    for epoch in range(1000):
        train_one_epoch(keyframe_net, optimizer, train_loader, lamdas, epoch, log_writer)
        loss = validate(keyframe_net,
                        valid_loader,
                        lamdas,
                        log_writer,
                        global_step=(epoch + 1) * len(train_loader.dataset))
        scheduler.step(loss)

        test_bd_rate(keyframe_net, test_loader, results_anchor, log_writer, (epoch + 1) * len(train_loader.dataset))

        # Save model checkpoint if validation loss improved
        if loss < best_loss:
            torch.save(keyframe_net.state_dict(), Path(run_path) / "checkpoint.pth")

    log_writer.flush()
