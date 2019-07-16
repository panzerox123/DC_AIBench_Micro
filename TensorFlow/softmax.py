#!test tensorflow API -- tf.nn.softmax

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
x_size = 100
data["x"] = np.float32(np.random.rand(x_size))

#session
sess = tf.Session()
result = tf.nn.softmax(data["x"], dim=-1)
begin = clock()
sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


