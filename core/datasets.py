# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, preload_data=False, repeat=1):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.repeat = repeat
        # 内存缓冲功能
        self.preload_data = preload_data
        self.preloaded_images = []  # 存储预加载的图像数据
        self.preloaded_flows = []   # 存储预加载的光流数据
        self.preloaded_valids = []  # 存储预加载的有效性掩码
        
    def _preload_all_data(self):
        """预加载所有数据到内存中"""
        if not self.preload_data:
            return
            
        print(f"正在预加载 {len(self.image_list)} 个样本到内存中...")
        
        for i in range(len(self.image_list)):
            # 加载图像
            img1 = frame_utils.read_gen(self.image_list[i][0])
            img2 = frame_utils.read_gen(self.image_list[i][1])
            
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            
            # 处理灰度图像
            if len(img1.shape) == 2:
                img1 = np.tile(img1[...,None], (1, 1, 3))
                img2 = np.tile(img2[...,None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
                
            self.preloaded_images.append((img1, img2))
            
            # 加载光流数据
            if i < len(self.flow_list):
                valid = None
                if self.sparse:
                    flow, valid = frame_utils.readFlowKITTI(self.flow_list[i])
                else:
                    flow = frame_utils.read_gen(self.flow_list[i])
                    
                flow = np.array(flow).astype(np.float32)
                self.preloaded_flows.append(flow)
                self.preloaded_valids.append(valid)
            else:
                self.preloaded_flows.append(None)
                self.preloaded_valids.append(None)
                
            if (i + 1) % 100 == 0:
                print(f"已预加载 {i + 1}/{len(self.image_list)} 个样本")
                
        print(f"预加载完成！共加载 {len(self.preloaded_images)} 个样本")

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        
        # 从预加载的内存数据中获取或从磁盘读取
        if self.preload_data and index < len(self.preloaded_images):
            # 从内存中获取预加载的数据
            img1, img2 = self.preloaded_images[index]
            img1 = img1.copy()  # 复制数据以避免修改原始缓存
            img2 = img2.copy()
            
            if index < len(self.preloaded_flows) and self.preloaded_flows[index] is not None:
                flow = self.preloaded_flows[index].copy()
                valid = self.preloaded_valids[index].copy() if self.preloaded_valids[index] is not None else None
            else:
                # 如果光流数据未预加载，从磁盘读取
                valid = None
                if self.sparse:
                    flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
                else:
                    flow = frame_utils.read_gen(self.flow_list[index])
                flow = np.array(flow).astype(np.float32)
        else:
            # 从磁盘读取数据（原始方式）
            #print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            valid = None
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[...,None], (1, 1, 3))
                img2 = np.tile(img2[...,None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list) * self.repeat
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/redpine/share2/dataset/MPI-Sintel-complete/', dstype='clean', preload_data=False, repeat=5):
        super(MpiSintel, self).__init__(aug_params, preload_data=preload_data, repeat=repeat)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
        
        # 如果启用预加载，则预加载所有数据
        if self.preload_data:
            self._preload_all_data()
class MpiSintel_val(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/redpine/share2/dataset/MPI-Sintel-complete/', dstype='clean', repeat=1):
        super(MpiSintel_val, self).__init__(aug_params, repeat=repeat)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in ['ambush_2','bamboo_2','cave_2','market_2','shaman_2','temple_2']:
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        #things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        train_dataset = 100*sintel_clean + 100*sintel_final 
        # if TRAIN_DS == 'C+T+K+S+H':
        #     kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        #     hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        #     train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        # elif TRAIN_DS == 'C+T+K/S':
        #     train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

