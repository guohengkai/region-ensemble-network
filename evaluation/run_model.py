import os
import sys
from hand_model import HandModel
import util


def print_usage():
    print('usage: {} icvl/nyu model_prefix out_file base_dir'.format(sys.argv[0]))
    exit(-1)


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
    util.save_results(results, out_file)

if __name__ == '__main__':
    main()
