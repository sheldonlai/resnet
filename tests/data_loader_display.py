import matplotlib.pyplot as plt
import numpy as np

from data_loader import CifarDataLoader
from util.image_utils import pad_images, random_corner_crop_flip

dl = CifarDataLoader()
data, _ = dl.get_data()

num_sample = 4
data_batch = data[:num_sample]
data_batch = pad_images(data_batch)
data_batch = random_corner_crop_flip(data_batch, crop_dim=[32, 32, 3])
data_batch = np.concatenate((data_batch, data[:num_sample]), axis=2)

temp = data_batch[1]
for image in data_batch[1:]:
    temp = np.concatenate((temp, image), axis=0)

data_batch = temp
data_batch = (data_batch + 1) * 127.5

plt.imshow(data_batch)


plt.show()
print("done")
