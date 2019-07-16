#!test tensorflow API -- tf.add

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

#input_y
y_size = 100
data["y"] = np.float32(np.random.rand(y_size))


#session
sess = tf.Session()
result = tf.add(data["x"], data["y"])
begin = clock()
sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


