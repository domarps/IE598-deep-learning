import numpy as np
from scipy import ndimage
import time

import os

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

#load CIFAR10 data
import h5py
CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
x_train = np.float32(CIFAR10_data['X_train'][:] )
y_train = np.int32(np.array(CIFAR10_data['Y_train'][:]))
x_test = np.float32(CIFAR10_data['X_test'][:] )
y_test = np.int32( np.array(CIFAR10_data['Y_test'][:]  ) )


aug_X_train = []
aug_Y_train = []
for xt in  x_train:

	temp_image = np.rollaxis(xt, 0, 3)
	temp_image_flr = np.fliplr(temp_image)  # flip
	temp_image_con = (temp_image - temp_image.min()) / (temp_image.max() - temp_image.min())  # normalize
	temp_image_med = ndimage.median_filter(temp_image, 2)  # median filter
	print xt.shape
	print temp_image.shape
	print temp_image_flr.shape
	print temp_image_med.shape
	print temp_image_con.shape
	aug_X_train.append(temp_image)
	aug_X_train.append(temp_image_flr)
	aug_X_train.append(temp_image_con)
	aug_X_train.append(temp_image_med)

for yt in y_train:
	aug_Y_train.append(yt)
	aug_Y_train.append(yt)
	aug_Y_train.append(yt)
	aug_Y_train.append(yt)

a_X_train = np.array(aug_X_train)
a_Y_train = np.array(aug_Y_train)

x_train = a_X_train
y_train = a_Y_train
CIFAR10_data.close()

