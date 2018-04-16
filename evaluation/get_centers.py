import os
import sys
import cv2
import util
import numpy as np


def print_usage():
    print('usage: {} icvl/nyu/msra base_dir out_file'.format(sys.argv[0]))
    exit(-1)


def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for result in results:
            for j in range(result.shape[0]):
                for k in range(result.shape[1]):
                    f.write('{:.3f} '.format(result[j, k]))
            f.write('\n')

def main():
    if len(sys.argv) < 4:
        print_usage()

    dataset = sys.argv[1]
    base_dir = sys.argv[2]
    out_file = sys.argv[3]
    names = util.load_names(dataset)
    centers = []
    for idx, name in enumerate(names):
        if dataset == 'nyu':  # use synthetic image to compute center
            name = name.replace('depth', 'synthdepth')
        img = util.load_image(dataset, os.path.join(base_dir, name))
        if dataset == 'icvl':
            center = util.get_center(img, upper=500, lower=0)
        elif dataset == 'nyu':
            center = util.get_center(img, upper=1300, lower=500)
        elif dataset == 'msra':
            center = util.get_center(img, upper=1000, lower=10)
        centers.append(center.reshape((1, 3)))
        if idx % 500 == 0:
            print('{}/{}'.format(idx + 1, len(names)))
    util.save_results(centers, out_file)

if __name__ == '__main__':
    main()
