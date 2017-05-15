import argparse
import os
import sys
import cv2
from hand_model import HandModel
import util
import numpy as np


def print_usage():
    print('usage: {} icvl/nyu model_prefix in_file out_file'.format(sys.argv[0]))
    exit(-1)


def get_center(img, upper=650, lower=1):
    centers = np.array([0.0, 0.0, 0.0])
    count = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] <= upper and img[y, x] >= lower:
                centers[0] += x
                centers[1] += y
                centers[2] += img[y, x]
                count += 1
    if count:
        centers /= count
    return centers


def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for result in results:
            for j in range(result.shape[0]):
                for k in range(result.shape[1]):
                    f.write('{:.3f} '.format(result[j, k]))
            f.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_model', help='the dataset type for model')
    parser.add_argument('model_prefix', help='the model prefix')
    parser.add_argument('base_dir', help='the base directory for image')
    parser.add_argument('in_file', default=None, help='input image list')
    parser.add_argument('out_file', default=None, help='output file for pose')
    parser.add_argument('--dataset_image', default=None,
            help='the dataset type for loading images, use the same as dataset_model when empty')
    parser.add_argument('--is_flip', action='store_true', help='flip the input')
    parser.add_argument('--upper', type=int, default=700, help='upper value for segmentation')
    parser.add_argument('--lower', type=int, default=1, help='lower value for segmentation')
    parser.add_argument('--fx', type=float, default=371.62, help='fx')
    parser.add_argument('--fy', type=float, default=370.19, help='fy')
    parser.add_argument('--ux', type=float, default=256, help='ux')
    parser.add_argument('--uy', type=float, default=212, help='uy')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_model = args.dataset_model
    dataset_image = args.dataset_image
    if dataset_image is None:
        dataset_image = dataset_model

    hand_model = HandModel(dataset_model, args.model_prefix,
            lambda img: get_center(img, lower=args.lower, upper=args.upper),
            param=(args.fx, args.fy, args.ux, args.uy))
    with open(args.in_file) as f:
        names = [line.strip() for line in f]
    results = hand_model.detect_files(args.base_dir, names, dataset=dataset_image,
            is_flip=args.is_flip)
    save_results(results, args.out_file)


if __name__ == '__main__':
    main()
