import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import pyrealsense as pyrs
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../evaluation'))
from evaluation.hand_model import HandModel
import evaluation.util as util
from evaluation.util import get_center
    
def read_frame_from_device(dev):
    dev.wait_for_frame()
    img_rgb = dev.colour
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    depth = dev.depth * dev.depth_scale * 1000
    return depth, img_rgb

def show_results(img, results, dataset):
    img = np.minimum(img, 1500)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_show = util.draw_pose(dataset, img, results)
    return img_show

def main():
    # intrinsic paramters of Intel Realsense SR300
    fx, fy, ux, uy = 463.889, 463.889, 320, 240
    # paramters
    dataset = 'icvl'
    model = 'ren_9x6x6'
    lower_ = 1
    upper_ = 650

    # init hand pose estimation model
    hand_model = HandModel(dataset, model,
        lambda img: get_center(img, lower=lower_, upper=upper_),
        param=(fx, fy, ux, uy), use_gpu=False)
    # realtime hand pose estimation loop
    with pyrs.Service() as serv:
        with serv.Device() as dev:
            while True:
                dev.apply_ivcam_preset(0)
                dev.wait_for_frames()
                depth = dev.depth * dev.depth_scale * 1000
                # preprocessing depth
                depth[depth == 0] = depth.max()
                depth = depth[:, ::-1]  # flip
                # get hand pose
                results = hand_model.detect_image(depth)
                img_show = show_results(depth, results, dataset)
                cv2.imshow('result', img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
if __name__ == '__main__':
    main()
