#!test tensorflow API -- tf.nn.relu

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

#input_x
input_batch = int(sys.argv[1])
x_size = int(sys.argv[2])
input_channels = int(sys.argv[3])
#data["x"] = np.float32(np.random.rand(x_size))

#session
sess = tf.Session()
#result = tf.nn.relu(data["x"])
result = tf.nn.relu(np.float32(np.random.rand(int(x_size*x_size/56*input_channels))))
begin = clock()
for i in range(input_batch*input_batch*56):
    sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


