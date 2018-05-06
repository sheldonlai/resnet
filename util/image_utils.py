import math

import numpy as np
import tensorflow as tf


def add_gaussian_noise(images):
    gaussian_noise_imgs = []
    row, col, chn = images[0].shape
    # Gaussian distribution parameters

    for img in images:
        gaussian = np.random.normal(0, 0.1, (row, col, chn)).astype(np.float32)
        gaussian_img = img + gaussian
        gaussian_noise_imgs.append(gaussian_img)
    return np.clip(np.array(gaussian_noise_imgs), -1, 1)


def add_gaussian_noise_image(image):
    row, col, chn = image.shape
    # Gaussian distribution parameters

    gaussian = np.random.normal(0, 0.1, (row, col, chn)).astype(np.float32)
    gaussian_img = np.clip(image + gaussian, -1, 1)

    return gaussian_img


def rotate(images):
    with tf.Session() as sess:
        rotate = np.random.uniform(-1, 1, (len(images)))
        rotate = rotate * math.pi / 8  # max 27.5 degrees
        rotated_images = sess.run(tf.contrib.image.rotate(images, rotate))

    return rotated_images


def pad_images(images, padding=4, constant_values=-1):
    with tf.Session() as sess:
        padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
        res = sess.run(tf.pad(images, padding, mode='CONSTANT', constant_values=constant_values))
    return res


def random_crop(images, crop_dim=(32, 32, 3)):
    with tf.Session() as sess:
        res = sess.run([tf.random_crop(image, crop_dim) for image in images])
    return res


def random_corner_crop_flip(images, crop_dim=(32, 32, 3)):
    n, h, w, _ = images.shape
    nums = np.random.randint(0, h - crop_dim[0], (n, 2))
    flip_nums = np.random.randint(0, 2, n)
    # noise_nums = np.random.randint(0, 2, n)
    cropped = []
    for i, (num1, num2) in enumerate(nums):
        temp_img = images[i, num1:num1+crop_dim[0], num2:num2+crop_dim[1]]
        # if num == 0:
        #     temp_img = images[i, :crop_dim[0], :crop_dim[1]]
        # elif num == 1:
        #     temp_img = images[i, h - crop_dim[0]:, :crop_dim[1]]
        # elif num == 2:
        #     temp_img = images[i, h - crop_dim[0]:, w - crop_dim[1]:]
        # elif num == 3:
        #     temp_img = images[i, :crop_dim[0], w - crop_dim[1]:]
        # else:
        #     temp_img = images[i, 4:crop_dim[0]+4, 4:crop_dim[1]+4]

        if flip_nums[i] == 0:
            temp_img = np.flip(temp_img, 1)

        # if noise_nums[i] == 0:
        #     temp_img = add_gaussian_noise_image(temp_img)

        cropped.append(temp_img)

    return np.array(cropped)

