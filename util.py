from __future__ import division
from __future__ import print_function
import os, glob, shutil, math
import tensorflow as tf
import numpy as np
from PIL import Image

import config


def exists_or_mkdir(path, need_remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif need_remove:
        shutil.rmtree(path)
        os.makedirs(path)
    return None


def save_list(save_path, data_list):
    n = len(data_list)
    with open(save_path, 'w') as f:
        f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None


def save_images_from_batch(img_batch, save_dir, init_no):
    if img_batch.shape[-1] == 3:
        ## rgb color image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, :]+1)+0.5).astype(np.uint8))
            image.save(os.path.join(save_dir, 'restored_rgb_%05d.png' % (init_no + i)), 'PNG')
    else:
        ## single-channel gray image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, 0]+1)+0.5).astype(np.uint8))
            image.save(os.path.join(save_dir, 'inverted_gray_%05d.png' % (init_no + i)), 'PNG')
    return None


def compute_color_psnr(im_batch1, im_batch2):
    """ Compute color psnr in 3 channels
    Args:
        im_batch1 (numpy array): numpy array of source image
        im_batch2 (numpy array): numpy array of target image
    Return:
        psnr (float)
    """
    mean_psnr = 0
    im_batch1 = im_batch1.squeeze()
    im_batch2 = im_batch2.squeeze()
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        # print(im1.shape)
        psnr1 = calc_psnr(im1[:,:,0], im2[:,:,0])
        psnr2 = calc_psnr(im1[:,:,1], im2[:,:,1])
        psnr3 = calc_psnr(im1[:,:,2], im2[:,:,2])
        mean_psnr += (psnr1+psnr2+psnr3) / 3.0
    return mean_psnr/num


def measure_psnr(im_batch1, im_batch2):
    """ measure psnr
    Args:
        im_batch1 (numpy array): numpy array of source image
        im_batch2 (numpy array): numpy array of target image
    Return:
        psnr (float)
    """
    mean_psnr = 0
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        psnr = calc_psnr(im1, im2)
        mean_psnr += psnr
    return mean_psnr/num


def calc_psnr(im1, im2):
    '''
    Notice: Pixel value should be convert to [0,255]
    '''
    if im1.shape[-1] != 3:
        g_im1 = im1.astype(np.float32)
        g_im2 = im2.astype(np.float32)
    else:
        g_im1 = np.array(Image.fromarray(im1.astype('uint8')).convert('L'), np.float32)
        g_im2 = np.array(Image.fromarray(im2.astype('uint8')).convert('L'), np.float32)

    mse = np.mean((g_im1 - g_im2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def gaussian_noise_layer(input_layer, mean=0.0, stddev=1.0):
    ''' Add noise to weights of model
    Args:
        w (tf.Varaible): weights
        mean (float): mean
        stddev (float): standard deviation
    Returns:
        updated values of the weights
    '''
    variables_shape = tf.shape(input_layer)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    return input_layer + noise
    # return tf.assign_add(input_layer, noise)

def generate_rgb_gradient_image(img_shape, img_dir):
    import math
    from PIL import Image
    im = Image.new('RGB', img_shape)
    ld = im.load()

    def gaussian(x, a, b, c, d=0):
        return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

    for x in range(im.size[0]):
        r = int(gaussian(x, 158.8242, 201, 87.0739) + gaussian(x, 158.8242, 402, 87.0739))
        g = int(gaussian(x, 129.9851, 157.7571, 108.0298) + gaussian(x, 200.6831, 399.4535, 143.6828))
        b = int(gaussian(x, 231.3135, 206.4774, 201.5447) + gaussian(x, 17.1017, 395.8819, 39.3148))
        for y in range(im.size[1]):
            ld[x, y] = (r, g, b)

    im.save(os.path.join(img_dir, 'color_gradient'), 'PNG')
