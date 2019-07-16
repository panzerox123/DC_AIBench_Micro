#!test tensorflow API -- tf.nn.max_pool

import sys
import os
import tensorflow as tf
import numpy as np
import copy
from time import clock
import time

#CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

data = dict()

#value
value_batch = int(sys.argv[1])
value_height = int(sys.argv[2])
value_width = int(sys.argv[2])
value_channels = int(sys.argv[3])
#data["value"] = np.float32(np.random.rand(value_batch, value_height, value_width, value_channels))

#ksize
data["ksize"] = [1, 2, 2, 1]

#strides
data["strides"] = [1, 2, 2, 1]


#session
sess = tf.Session()
result = tf.nn.max_pool(np.float32(np.random.rand(value_batch, value_height, value_width, value_channels)), data["ksize"], data["strides"], padding='SAME', data_format='NHWC')
begin = clock()
for i in range(int(sys.argv[1])):
    sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


