from IFNET_m import IFNet_m_flow
import torch
import numpy as np


if __name__ == "__main__":
    model = IFNet_m_flow()

    path = r'/home/redpine/share2/code_frame_interp/ECCV2022-RIFE-main/train_log_bothdata/flownet_latest.pkl'
    check_point = torch.load(path)
    ind = 0
    for key in check_point:
        print(ind, key, check_point[key].shape==model.state_dict()[key].shape)
        ind += 1
    # ind = 0
    # for key in model.state_dict():
    #     print(ind, key, model.state_dict()[key].shape)
    #     ind += 1
    
