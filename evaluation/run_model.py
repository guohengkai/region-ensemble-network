import os
import sys
from hand_model import HandModel
import util


def print_usage():
    print('usage: {} icvl/nyu model_prefix out_file base_dir'.format(sys.argv[0]))
    exit(-1)


def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                for k in range(results.shape[2]):
                    f.write('{} '.format(results[i, j, k]))
            f.write('\n')

def main():
    if len(sys.argv) < 5:
        print_usage()

    dataset = sys.argv[1]
    model = sys.argv[2]
    out_file = sys.argv[3]
    base_dir = sys.argv[4]

    hand_model = HandModel(dataset, model)
    names = util.load_names(dataset)
    centers = util.load_centers(dataset)
    results = hand_model.detect_files(base_dir, names, centers)
    save_results(results, out_file)

if __name__ == '__main__':
    main()
