import threading
from queue import Queue
import numpy as np

from util.image_utils import pad_images, random_crop, random_corner_crop_flip


# I didn't address the potential dead lock that I didn't think would play a factor in normal training
class DataQueue(object):
    def __init__(self, data, labels, batch_size, capacity=16, threads=1, data_aug=True):
        """

        :param data: ndarray of 4D
        :param capacity:
        """
        self.data_shape = data.shape
        self.capacity = capacity
        self.q = Queue(capacity)

        self.batch_size = batch_size
        self._started = False
        self._num_threads = threads
        self._threads = []
        self._padded_data = data
        self._labels = labels

        self._data_aug = data_aug

    def push(self, data):
        self.q.put(data, block=True, timeout=None)

    def pop(self):
        # TODO: a timeout might be good
        if self.q.qsize() <= self.q.maxsize // 2:
            print('current has %d batches in queue' % self.q.qsize())
        return self.q.get(block=True)

    def size(self):
        return self.q.qsize()

    def _populate_q(self):
        i = np.arange(self.data_shape[0])
        np.random.shuffle(i)
        cur = 0
        while self._started:
            start = cur*self.batch_size
            end = (cur + 1) * self.batch_size
            if end > self.data_shape[0]:
                l = np.concatenate((i[start:self.data_shape[0]], i[:end - self.data_shape[0]]), axis=0)
                data_batch = self._padded_data[l]
                if self._data_aug:
                    data_batch = random_corner_crop_flip(data_batch)
                self.push((data_batch, self._labels[l]))
                i = np.arange(self.data_shape[0])
                np.random.shuffle(i)
                cur = 0
            else:
                data_batch = self._padded_data[i[start:end]]
                if self._data_aug:
                    data_batch = random_corner_crop_flip(data_batch)
                self.push((data_batch, self._labels[i[start:end]]))

            cur += 1

    def start(self):
        if not self._started:
            self._started = True
            self._threads = [threading.Thread(target=self._populate_q, args=(), daemon=True)
                             for i in range(self._num_threads)]
            for t in self._threads:
                t.start()
        else:
            print('queue is running already')

    def stop(self):
        self._started = False
        for t in self._threads:
            t.join()
        self._threads = []
