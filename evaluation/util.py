import cv2
import numpy as np


def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, len(positions[0]) / 3, 3))


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu', 'msra'])


def get_dataset_file(dataset):
    return 'labels/{}_test_label.txt'.format(dataset)


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.03, 587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120


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


def load_image(dataset, name, input_size=None, is_flip=False):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    if dataset == 'icvl':
        img = cv2.imread(name, 2)  # depth image
        img[img == 0] = img.max()  # invalid pixel
        img = img.astype(float)
    elif dataset == 'nyu':
        ori_img = cv2.imread(name)
        if input_size is not None:
            ori_img = cv2.resize(ori_img, (input_size, input_size))
        img = np.empty(ori_img.shape[:2], dtype=np.float32)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                img[r, c] = (ori_img[r, c, 1] << 8) + ori_img[r, c, 0]
    elif dataset == 'msra':  # TODO
        return None

    if input_size is not None:
        img = cv2.resize(img, (input_size, input_size))
    if is_flip:
        img[:, ::-1] = img
    return img


def load_names(dataset):
    with open('labels/{}_test_list.txt'.format(dataset)) as f:
        return [line.strip() for line in f]


def load_centers(dataset):
    with open('labels/{}_center.txt'.format(dataset)) as f:
        return np.array([map(float,
            line.strip().split()) for line in f])


def get_sketch_setting(dataset):
    if dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'nyu':
        return [(0, 1), (0, 2), (0, 5), (3, 4), (4, 5), (0, 7), (6, 7),
                (0, 9), (8, 9), (0, 11), (10, 11), (0, 13), (12, 13)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]

def draw_pose(dataset, img, pose):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 0, 255), 1)
    return img


def get_center(img, upper=650, lower=1):
    centers = np.array([0.0, 0.0, 300.0])
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
