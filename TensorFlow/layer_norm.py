#!test tensorflow API -- tf.contrib.layers.layer_norm

import os
import tensorflow as tf
import numpy as np
import copy
from time import clock
import time

#CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''

#inputs
inputs = tf.Variable(tf.random_normal([128, 32, 32, 64]))

#session
sess = tf.Session()

result = tf.contrib.layers.layer_norm(inputs)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

begin = clock()
sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
sess.close()


