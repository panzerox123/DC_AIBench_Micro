#!test tensorflow API -- tf.multiply

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

#input_A
height = int(sys.argv[2])*int(sys.argv[3])/(224/int(sys.argv[2]))
width = int(sys.argv[2])*int(sys.argv[3])/(224/int(sys.argv[2]))
#data["A"] = np.float32(np.random.rand(A_height, A_width))

#input_B
#B_height = int(sys.argv[2])*int(sys.argv[3])
#B_width = int(sys.argv[2])*int(sys.argv[3])
#data["B"] = np.float32(np.random.rand(B_height, B_width))


#session
sess = tf.Session()
result = tf.matmul(np.float32(np.random.rand(height, width)), np.float32(np.random.rand(height, width)))
begin = clock()
for i in range(int(sys.argv[1])):
    sess.run(result)
end = clock()
print("computings times: %f second" % (end - begin))
print(result)
sess.close()


