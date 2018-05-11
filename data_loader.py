import numpy as np

from util.utils import get_cifar10_all_data, \
    get_cifar10_test_data, get_padded_cifar10_data


class CifarDataLoader(object):

    def __init__(self, augmentation=False, normalize=False, partition=1, selection=0):
        """
        :param data_batch: data_batch is in range [0,9]
        :param scale: float in range [0 ,1.0] to scale image down
        """

        self.augmentation = augmentation
        self.normalize = normalize
        self.mean = None
        self.stddev = None

        self.partition = partition <= 1
        self.partition_num = partition
        self.selection = selection

    def get_data(self, selection=None):
        if not self.augmentation:
            m_data, labels = get_cifar10_all_data()
            m_data = (m_data / 127.5) - 1

            # mask = np.arange(0, len(m_data) // 10)
            mask = np.random.choice(len(m_data), len(m_data) // 10, replace=False)

            valid_data = m_data[mask]
            valid_labels = labels[mask]

            m_data = np.array([e for i, e in enumerate(m_data) if i not in mask])
            labels = np.array([e for i, e in enumerate(labels) if i not in mask])
        else:
            m_data, labels, valid_data, valid_labels = get_padded_cifar10_data()
            if self.normalize:
                self.mean = np.mean(m_data[:, 4:-4, 4:-4, :])
                self.stddev = np.std(m_data[:, 4:-4, 4:-4, :])
                print("data mean: %4.4f, std: %4.4f" % (float(self.mean), float(self.stddev)))

                m_data = (m_data - self.mean) / self.stddev

        if self.partition:
            if selection is None:
                selection = self.selection
            start = selection * len(m_data) // self.partition
            end = (selection + 1) * len(m_data) // self.partition
            if end > len(m_data):
                raise IndexError('partition end is out of bounds')
            m_data = m_data[start:end]

        return m_data, labels, valid_data, valid_labels

    def get_test_data(self):
        m_data, labels = get_cifar10_test_data()
        m_data = (m_data / 127.5) - 1

        if self.normalize:
            if self.mean is None:
                self.get_data()
            m_data = (m_data - self.mean) / self.stddev

        return m_data, labels
