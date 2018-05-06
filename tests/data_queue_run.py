import time

from data_loader import CifarDataLoader
from data_queue import DataQueue

dl = CifarDataLoader()
data, labels, _, _ = dl.get_data()
print('making data q')
dq = DataQueue(data, labels, 128, capacity=100, threads=16)
print('done making q')

start = time.time()
dq.start()
for i in range(6000):
    print('popping' + str(i))
    x, y = dq.pop()
    assert(x.shape[0] == 128), x.shape[0]
    assert(y.shape[0] == 128), y.shape[0]
print(time.time() - start)
dq.stop()
