#!test tensorflow API -- tf.nn.batch_normalization

import os
import tensorflow as tf
import numpy as np
import copy
from time import clock
import time

#CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

#image
image_batch = 128
image_height = 32
image_width = 32
image_channels = 64
image = tf.Variable(tf.random_normal([image_batch, image_height, image_width, image_channels]))
#image = np.float32(np.random.rand(image_batch, image_height, image_width, image_channels))

#axis
axis = list(range(len(image.get_shape()) - 1))

#mean, variance
mean, variance = tf.nn.moments(image, axis)

#offset
offset = tf.Variable(tf.random_normal(mean.get_shape()))

#scale
scale = tf.Variable(tf.random_normal(mean.get_shape()))

#variance_epsilon
variance_epsilon = 0.001


#session
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

result = tf.nn.batch_normalization(image, mean, variance, offset, scale, variance_epsilon)
begin = clock()
sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


