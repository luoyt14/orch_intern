import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor, water_row_anchor
import einops
from PIL import Image

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    elif cfg.dataset == "water":
        cls_num_per_lane = 64
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,cfg.num_lanes),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((640, 320)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    elif cfg.dataset == 'water':
        splits = ['test_only_ori.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/'+split),isrotate = True,img_transform = img_transforms) for split in splits]
        img_w, img_h = 480, 640
        row_anchor = water_row_anchor
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # print(split[:-3]+'avi')
        # vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            # vis = imgs.clone().detach()[0].numpy()
            # imean, istd = np.array((0.485, 0.456, 0.406)), np.array((0.229, 0.224, 0.225))
            # imean = einops.repeat(imean, 'c ->c w h', w=288, h=800)
            # istd = einops.repeat(istd, 'c ->c w h', w=288, h=800)
            # vis = vis * istd + imean
            # vis = vis.astype(np.uint8) * 255
            # vis = einops.rearrange(vis, 'c w h -> w h c')
            # vis = cv2.resize(vis, (img_w, img_h))
            # print(vis.shape)
            imgs = imgs.cuda()
            with torch.no_grad():
                out = net(imgs)

            col_sample = np.linspace(0, 320 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]


            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            # out_j = out_j[::-1, :, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc
            # print(out_j)

            # import pdb; pdb.set_trace()
            # vis = Image.open(os.path.join(cfg.data_root,names[0]))
            # if cfg.dataset == "water":
            #     vis = vis.transpose(Image.ROTATE_90)
            # vis = np.asarray(vis)
            # print(vis.shape)
            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            # print(vis.shape)
            if cfg.dataset == "water":
                vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_h, img_w, _ = vis.shape
                # print(f"img_w={img_w},img_h={img_h}")
            # print(row_anchor)
            # print(cls_num_per_lane)
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 320) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/640)) - 1 )
                            # print(int(out_j[k, i] * col_sample_w * img_w / 320))
                            # ppp = (int(out_j[k, i] * col_sample_w), int(row_anchor[cls_num_per_lane-1-k]))
                            vis = cv2.circle(vis,ppp,5,(0,255,0),-1)
                            # print(ppp[0])
            if cfg.dataset == "water":
                vis = cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE)
                # name = "/".join(names[0].split("/")[2:])
                # path = "/".join(names[0].split("/")[2])
                name = names[0]
                path = "/".join(names[0].split("/")[:-1])
            else:
                name = names[0]
                path = "/".join(names[0].split("/")[:-1])
            # print(name)
            save_path = os.path.join(cfg.data_root, "vis/" + path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # print(save_path)
            cv2.imwrite(os.path.join(cfg.data_root, "vis/" + name), vis)
            # vout.write(vis)
            # break
        
        # vout.release()
