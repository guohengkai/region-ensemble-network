import os
import sys
import cv2
import numpy as np
from util import draw_pose, get_positions, load_image, load_names


def print_usage():
    print('usage: {} icvl/nyu in_file base_dir'.format(sys.argv[0]))
    exit(-1)


def show_pose(dataset, base_dir, outputs):
    names = load_names(dataset)
    assert len(names) == outputs.shape[0]
    for idx, (name, pose) in enumerate(zip(names, outputs)):
        img = load_image(dataset, os.path.join(base_dir, name))
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = draw_pose(dataset, img, pose) 
        cv2.imshow('result', img)
        ch = cv2.waitKey(40)
        if ch == ord('q'):
            break


def main():
    if len(sys.argv) < 4:
        print_usage()

    dataset = sys.argv[1]
    in_file = sys.argv[2]
    base_dir = sys.argv[3]

    outputs = get_positions(in_file)
    show_pose(dataset, base_dir, outputs)


if __name__ == '__main__':
    main()
