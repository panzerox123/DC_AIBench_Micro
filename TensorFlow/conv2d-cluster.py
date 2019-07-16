#!test tensorflow API -- tf.nn.conv2d
# -*- coding:utf-8 -*-

import sys
import os
import glob
import tensorflow as tf
import numpy as np
import copy
from time import clock
import time

#CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

data = dict()

#input
input_batch = int(sys.argv[1])
input_height = int(sys.argv[2])
input_width = int(sys.argv[2])
input_channels = int(sys.argv[3])
data["input"] = np.float32(np.random.rand(input_batch, input_height, input_width, input_channels))

#filter
filter_height = int(sys.argv[4])
filter_width = int(sys.argv[4])
filter_in_channels = int(sys.argv[3])
filter_out_channels = int(sys.argv[3])
data["filter"] = np.float32(np.random.rand(filter_height, filter_width, filter_in_channels, filter_out_channels))

#strides
data["strides"] = [1, 1, 1, 1]


#session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

result = tf.nn.conv2d(data["input"], data["filter"], data["strides"], padding='SAME')
begin = clock()
for i in range(int(sys.argv[1])):
    sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


