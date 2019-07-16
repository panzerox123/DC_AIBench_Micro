#!test tensorflow API -- tf.nn.dropout

import os
import tensorflow as tf
import numpy as np
import copy
from time import clock
import time

#CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

#keep_prob
keep_prob = tf.placeholder(tf.float32)

#inputs
inputs = np.float32(np.random.rand(100, 100))

#outputs
outputs = tf.nn.dropout(inputs, keep_prob, noise_shape=None, seed=None)

#session
sess = tf.Session()
begin = clock()
sess.run(outputs, feed_dict = {keep_prob : 0.4})
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


