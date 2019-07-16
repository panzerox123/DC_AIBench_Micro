#!test tensorflow API -- tf.nn.conv2d_transpose

import os
import tensorflow as tf
import numpy as np
import copy
from time import clock
import time

#CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

data = dict()

#input
input_batch = 5
input_height = 100
input_width = 100
input_channels = 3
data["input"] = np.float32(np.random.rand(input_batch, input_height, input_width, input_channels))

#filter
filter_height = 3
filter_width = 3
filter_in_channels = 3
filter_out_channels = 3
data["filter"] = np.float32(np.random.rand(filter_height, filter_width, filter_in_channels, filter_out_channels))

#output_shape
data["output_shape"] = [5, 100, 100, 3]

#strides
data["strides"] = [1, 1, 1, 1]


#session
sess = tf.Session()
result = tf.nn.conv2d_transpose(data["input"], data["filter"], data["output_shape"], data["strides"], padding='SAME')
begin = clock()
sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
print result
sess.close()


