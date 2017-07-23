import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from util import get_errors


def print_usage():
    print('usage: {} icvl/nyu/msra in_file'.format(sys.argv[0]))
    exit(-1)


def draw_error(dataset, errs):
    mean_errs = np.mean(errs, axis=0)
    mean_errs = np.append(mean_errs, np.mean(mean_errs))
    print('mean error: {:.2f}mm'.format(mean_errs[-1]))

    if dataset == 'icvl':
        joint_idx = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]
        names = ['Palm', 'Thumb.R', 'Thumb.T', 'Index.R', 'Index.T', 'Mid.R', 'Mid.T', 'Ring.R', 'Ring.T', 'Pinky.R', 'Pinky.T', 'Mean']
    elif dataset == 'nyu':
        joint_idx = [0, 1, 2, 5, 3, 13, 12, 11, 10, 9, 8, 7, 6, 14];
        names = ['Palm', 'Wrist1', 'Wrist2', 'Thumb.R', 'Thumb.T', 'Index.R', 'Index.T', 'Mid.R', 'Mid.T', 'Ring.R', 'Ring.T', 'Pinky.R', 'Pinky.T', 'Mean'];
    elif dataset == 'msra':
        joint_idx = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21]
        names = ['Wrist', 'Index.R', 'Index.T', 'Mid.R', 'Mid.T', 'Ring.R', 'Ring.T', 'Pinky.R', 'Pinky.T', 'Thumb.R', 'Thumb.T', 'Mean']

    x = np.arange(len(joint_idx))
    plt.figure()
    plt.bar(x, mean_errs[joint_idx])
    plt.xticks(x + 0.5, names, rotation='vertical')
    plt.ylabel('Mean Error (mm)')
    plt.grid(True)


def draw_map(errs):
    thresholds = np.arange(0, 85, 5)
    results = np.zeros(thresholds.shape)
    for idx, th in enumerate(thresholds):
        results[idx] = np.where(np.max(errs, axis=1) <= th)[0].shape[0] * 1.0 / errs.shape[0]

    plt.figure()
    plt.plot(thresholds, results)
    plt.xlabel('Distance Threshold (mm)')
    plt.ylabel('Fraction of frames within distance')
    plt.grid(True)
    

def main():
    if len(sys.argv) < 3:
        print_usage()

    dataset = sys.argv[1]
    in_file = sys.argv[2]

    errs = get_errors(dataset, in_file)
    draw_error(dataset, errs)
    draw_map(errs)
    plt.show()


if __name__ == '__main__':
    main()
