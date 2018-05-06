import numpy as np

from util.utils import get_cifar10_all_data, \
    get_cifar10_test_data, get_padded_cifar10_data


class CifarDataLoader(object):

    def __init__(self, augmentation=False):
        """
        :param data_batch: data_batch is in range [0,9]
        :param scale: float in range [0 ,1.0] to scale image down
        """

        self.augmentation = augmentation

    def get_data(self):
        if not self.augmentation:
            m_data, labels = get_cifar10_all_data()
            m_data = (m_data / 127.5) - 1

            # mask = np.arange(0, len(m_data) // 10)
            mask = np.random.choice(len(m_data), len(m_data) // 10, replace=False)

            valid_data = m_data[mask]
            valid_labels = labels[mask]

            m_data = np.array([e for i, e in enumerate(m_data) if i not in mask])
            labels = np.array([e for i, e in enumerate(labels) if i not in mask])

            return m_data, labels, valid_data, valid_labels
        else:
            return get_padded_cifar10_data()

    def get_test_data(self):
        m_data, labels = get_cifar10_test_data()
        m_data = (m_data / 127.5) - 1
        return m_data, labels
