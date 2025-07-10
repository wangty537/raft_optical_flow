import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core import datasets
from core.utils import flow_viz
from core.utils import frame_utils

from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate
from tqdm import tqdm

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=8):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel_val(split='training', dstype=dstype)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results
@torch.no_grad()
def validate_sintel_liteflownet3(model):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel_val(split='training', dstype=dstype)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
           
            image1 = image1.unsqueeze(0) / 255
            image2 = image2.unsqueeze(0) / 255


            # 加载图像
          
            img1 = image1 # rgb 13hw 0-1
            img2 = image2
            
            # 确保两张图像尺寸相同
            if img1.shape != img2.shape:
                raise ValueError(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")
            
            img1 = img1.to(device)
            img2 = img2.to(device)
            flow_gt = flow_gt.to(device)
            
            
           
            # 估计光流
           
            with torch.no_grad():
                # 准备输入 - 正确的格式是 [batch, 2, 3, H, W]
                images = torch.stack([img1, img2], dim=1)  # [1, 2, 3, H, W]
                input_dict = {'images': images}
                
                # 前向推理
                output = model(input_dict)
                if isinstance(output, dict):
                    if 'flows' in output:
                        flow = output['flows'][0, 0]  # 取第一个时间步的光流
                    elif 'flow' in output:
                        flow = output['flow']
                    else:
                        flow = list(output.values())[0]  # 取第一个输出
                else:
                    flow = output
    

          
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
           
            epe_list.append(epe.view(-1).cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model',default='raft-small.pth', type=str, help="restore checkpoint")
    parser.add_argument('--model',default='/home/redpine/share11/code/RAFT-master/liteflownet3s-sintel-89793e34.ckpt', type=str, help="restore checkpoint")
    parser.add_argument('--dataset', default='sintel', help="dataset for evaluation")
    parser.add_argument('--small', default=True, help='use small model')
    parser.add_argument('--mixed_precision', default=False, help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))
    from liteflownet3_simple import liteflownet3s
    model = liteflownet3s()
    
    # 加载权重
    model_path = args.model
    if model_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 移除可能的前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        # 普通PyTorch权重
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            #validate_sintel(model.module, iters=32) 
            # Validation (clean) EPE: 2.086691, 1px: 0.854195, 3px: 0.936246, 5px: 0.955037
            # Validation (final) EPE: 3.682227, 1px: 0.796726, 3px: 0.886122, 5px: 0.912843

            validate_sintel_liteflownet3(model) 
            #Validation (clean) EPE: 2.243293, 1px: 0.845348, 3px: 0.929001, 5px: 0.949448
            #Validation (final) EPE: 4.046093, 1px: 0.784667, 3px: 0.877854, 5px: 0.904865

            # sintel ft之后, iter=8
            # Validation (clean) EPE: 1.553040, 1px: 0.869485, 3px: 0.941848, 5px: 0.959129
            # Validation (final) EPE: 2.631492, 1px: 0.818702, 3px: 0.901543, 5px: 0.926121

            # liteflownet3
            # Validation (clean) EPE: 1.755342, 1px: 0.888644, 3px: 0.946157, 5px: 0.960402
            # Validation (final) EPE: 2.400769, 1px: 0.828982, 3px: 0.907415, 5px: 0.931669
        elif args.dataset == 'kitti':
            validate_kitti(model.module)


