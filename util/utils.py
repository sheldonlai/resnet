import os
import pickle

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np

from util.downloader import maybe_download_and_extract
from util.image_utils import pad_images


def write_image_matrix(array, name, dst_folder=None):
    if dst_folder is None:
        dst_folder = './image_out/'
    else:
        dst_folder = os.path.join(os.getcwd(), dst_folder)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    imageio.imwrite(os.path.join(dst_folder, name + '.png'), array)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def do_pickle(file, obj):
    with open(file, 'wb') as fo:
        pickle.dump(obj, fo, protocol=pickle.HIGHEST_PROTOCOL)
    return dict


def _maybe_download_cifar10_data():
    path = './data/cifar10'

    if not os.path.exists(path):
        os.makedirs(path)

    maybe_download_and_extract(path)


def get_cifar10_dict():
    data = {}
    path = os.path.join(os.getcwd(), './data/cifar10/cifar-10-batches-py')

    _maybe_download_cifar10_data()

    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]

    for file_path in file_names:
        images, labels = cifar10_batch_to_matrix(unpickle(file_path))
        for i in range(len(images)):
            if str(labels[i]) in data:
                data[str(labels[i])].append(images[i])
            else:
                data[str(labels[i])] = [images[i]]

    for k, val in data.items():
        data[k] = np.array(val)
    return data


def get_cifar10_train_and_valid(standardize=True):
    data = get_cifar10_dict()
    m_data = None
    labels = None

    valid_data = None
    valid_labels = None

    for k, val in data.items():
        if standardize:
            val = convert_uint_to_tanh_range(np.array(val))
        mask = dict(zip(np.random.choice(5000, 500, replace=False), np.ones(500)))
        label_int = int(float(k))
        l = np.ones(4500) * label_int
        v_l = np.ones(500) * label_int

        valid = val[:500]
        train = val[500:]


        if m_data is None:
            valid_data = valid  # np.array([e for i, e in enumerate(val) if i in mask])
            valid_labels = v_l

            m_data = train  # np.array([e for i, e in enumerate(val) if i not in mask])
            labels = l

        else:
            valid_data = np.vstack((valid_data, valid))
            valid_labels = np.concatenate((valid_labels, v_l))

            m_data = np.vstack([m_data, train])
            labels = np.concatenate((labels, l))

    assert len(m_data) == 45000
    assert len(labels) == 45000
    assert len(valid_data) == 5000
    assert len(valid_labels) == 5000

    # i = 0
    # while i < 5000:
    #     print('image at i=%d is of class %d' % (i, valid_labels[i]))
    #     sample = valid_data[i]
    #     sample = (sample + 1) / 2
    #
    #     plt.imshow(sample)
    #
    #     plt.show()
    #     i += 500

    return m_data, labels.astype(np.int32), valid_data, valid_labels.astype(np.int32)


def cifar10_batch_to_matrix(data):
    m_data = np.swapaxes(np.swapaxes(data[b'data'].reshape([len(data[b'data']), 3, 32, 32]), 1, 3), 1, 2)
    m_labels = np.array(data[b'labels'], dtype=np.float32)
    return m_data, m_labels


def combine_image_arrays(data, img_dim):
    assert len(img_dim) == 2, "img_dim should be length 2 array"
    batch_size, x_dim, y_dim, channels = data.shape

    h, w = [round(e) for e in img_dim]
    # assert batch_size == h * w, "batch_size should equal img_dim[0] * img_dim[1]"

    matrix = np.array([])
    for i in range(h):
        row_array = [np.reshape(data[i * w + j], [x_dim, y_dim, channels]) for j in range(w)]
        row = np.concatenate(row_array, axis=1)
        if len(matrix) == 0:
            matrix = row
        else:
            matrix = np.vstack([matrix, row])
    return matrix.astype(np.uint8)


def convert_uint_to_tanh_range(images):
    return (images / 127.5) - 1


def convert_tanh_float_to_uint(images):
    return (images + 1) * 127.5


def get_cifar10_batch():
    data = []
    path = os.path.join(os.getcwd(), './data/cifar10/cifar-10-batches-py')
    _maybe_download_cifar10_data()
    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]
    for file_path in file_names:
        data.append(unpickle(file_path))
    return data, unpickle(os.path.join(path, 'test_batch'))


def get_cifar10_all_data():
    path = os.path.join(os.getcwd(), './data/cifar10/cifar-10-batches-py')
    _maybe_download_cifar10_data()
    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]
    m_data = None
    labels = None
    for name in file_names:
        data = unpickle(name)
        features, l = cifar10_batch_to_matrix(data)
        if m_data is None:
            m_data = features
            labels = l
        else:
            m_data = np.vstack((m_data, features))
            labels = np.concatenate((labels, l))
    return m_data, labels


def get_cifar10_test_data():
    path = os.path.join(os.getcwd(), './data/cifar10/cifar-10-batches-py')
    _maybe_download_cifar10_data()
    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "test_batch" in f]
    m_data = None
    labels = None
    for name in file_names:
        data = unpickle(name)
        features, l = cifar10_batch_to_matrix(data)
        if m_data is None:
            m_data = features
            labels = l
        else:
            m_data = np.vstack((m_data, features))
            labels = np.concatenate((labels, l))
    return m_data, labels


def get_padded_cifar10_data(shuffle=True):
    path = os.path.join(os.getcwd(), './data/cifar10_padded')
    valid_file = os.path.join(path, 'valid_batch')

    if not os.path.exists(path) or len(os.listdir(path)) == 0:
        if not os.path.exists(path):
            os.makedirs(path)

        m_data, labels, valid_data, valid_labels = get_cifar10_train_and_valid()

        m_data = pad_images(m_data)

        if shuffle:
            # shuffle here
            i = np.arange(m_data.shape[0])
            np.random.shuffle(i)

            m_data = m_data[i]
            labels = labels[i]

            i = np.arange(valid_data.shape[0])
            np.random.shuffle(i)

            valid_data = valid_data[i]
            valid_labels = valid_labels[i]

        store_size = len(m_data) // 10

        for i in range(10):
            start = i * store_size
            end = (i + 1) * store_size
            if end > len(m_data):
                end = len(m_data)
            store_obj = {'data': m_data[start:end], 'labels': labels[start:end]}
            do_pickle(os.path.join(path, 'data_batch' + str(i)), store_obj)

        # store valid dictionary
        valid_obj = {'data': valid_data, 'labels': valid_labels}
        do_pickle(valid_file, valid_obj)

        return m_data, labels, valid_data, valid_labels

    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]
    m_data = None
    labels = None
    for name in file_names:
        data = unpickle(name)
        features = data['data']
        l = data['labels']

        if m_data is None:
            m_data = features
            labels = l
        else:
            m_data = np.vstack((m_data, features))
            labels = np.concatenate((labels, l))

    obj = unpickle(valid_file)
    valid_data = obj['data']
    valid_labels = obj['labels']

    return m_data, labels, valid_data, valid_labels
