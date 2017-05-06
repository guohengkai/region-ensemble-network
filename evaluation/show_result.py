import os
import sys
import cv2
import numpy as np
import util

def print_usage():
    print('usage: {} icvl/nyu base_dir [in_file]'.format(sys.argv[0]))
    exit(-1)


def show_pose(dataset, base_dir, outputs):
    names = util.load_names(dataset)
    assert len(names) == outputs.shape[0]
    for idx, (name, pose) in enumerate(zip(names, outputs)):
        img = util.load_image(dataset, os.path.join(base_dir, name))
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = util.draw_pose(dataset, img, pose) 
        cv2.imshow('result', img)
        ch = cv2.waitKey(40)
        if ch == ord('q'):
            break


def main():
    if len(sys.argv) < 3:
        print_usage()

    if len(sys.argv) < 4:
        print('no input file, using ground truth')
    dataset = sys.argv[1]
    base_dir = sys.argv[2]
    in_file = sys.argv[3] if len(sys.argv) > 3 else util.get_dataset_file(dataset)

    outputs = util.get_positions(in_file)
    show_pose(dataset, base_dir, outputs)


if __name__ == '__main__':
    main()
