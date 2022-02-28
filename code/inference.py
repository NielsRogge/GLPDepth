'''
Doyeon Kim, 2022
'''

import os
import cv2
import numpy as np
from collections import OrderedDict
from torchvision import transforms

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    device = torch.device('cpu')

    if args.save_visualize:
        result_path = os.path.join(args.result_dir)
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)

    print("\n1. Define Model")
    model = GLPDepth(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir, map_location="cpu")
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n3. Inference")
    
    image = cv2.imread(args.image_path)  # [H x W x C] and C: BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # input size should be multiple of 32
    h, w, c = image.shape
    new_h, new_w = h // 32 * 32, w // 32 * 32
    image = cv2.resize(image, (new_w, new_h))
    pixel_values = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        pred = model(pixel_values)
    pred_d = pred['pred_d']
        
    if args.save_visualize:
        save_path = os.path.join(result_path, "prediction.png")
        pred_d_numpy = pred_d.squeeze().cpu().numpy()
        pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        cv2.imwrite(save_path, pred_d_color)

    print("Done")


if __name__ == "__main__":
    main()
