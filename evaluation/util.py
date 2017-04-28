import cv2
import numpy as np


def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, len(positions[0]) / 3, 3))


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu'])


def get_dataset_file(dataset):
    return 'labels/{}_test_label.txt'.format(dataset)


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.03, 587.07, 320, 240


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def get_errors(dataset, in_file):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    labels = get_positions(get_dataset_file(dataset))
    outputs = get_positions(in_file)
    params = get_param(dataset)
    labels = pixel2world(labels, *params)
    outputs = pixel2world(outputs, *params)
    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    return errors


def get_model(dataset, name='ren_4x6x6'):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    return ('models/deploy_{}_{}.prototxt'.format(dataset, name),
            'models/model_{}_{}.caffemodel'.format(dataset, name))


def load_image(dataset, name, input_size=None):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    if dataset == 'icvl':
        #img = cv2.imread(name, cv2.CV_LOAD_IMAGE_ANYDEPTH)
        img = cv2.imread(name, 2)
        if input_size is not None:
            img = cv2.resize(img, (input_size, input_size))
        return img.astype(float)
    elif dataset == 'nyu':
        ori_img = cv2.imread(name)
        if input_size is not None:
            ori_img = cv2.resize(ori_img, (input_size, input_size))
        depth_img = np.empty(ori_img.shape[:2], dtype=np.float32)
        for r in range(depth_img.shape[0]):
            for c in range(depth_img.shape[1]):
                depth_img[r, c] = (ori_img[r, c, 1] << 8) + ori_img[r, c, 0]
        return depth_img


def load_names(dataset):
    with open('labels/{}_test_list.txt'.format(dataset)) as f:
        return [line.strip() for line in f]


def load_centers(dataset):
    with open('labels/{}_center.txt'.format(dataset)) as f:
        return np.array([map(float,
            line.strip().split()) for line in f])
