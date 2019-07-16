#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import copy
from time import clock
import time

configure = dict()
configure['ksize'] = 3
configure['stride'] = 1
configure['conv_filters_out'] = 3
configure['conv_filters_in'] = 3
data_dict = dict()
data_dict["conv"] = np.float32(np.random.rand(configure['ksize'], configure['ksize'], configure['conv_filters_in'], configure['conv_filters_out']))

def get_conv_filter(name):
    return tf.constant(data_dict[name], name="filter")

def conv_layer(bottom):
	name = 'conv'
	with tf.variable_scope(name):
		filt = get_conv_filter(name)
		stride = [1, 1, 1, 1]
		conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
		return conv


x_data = np.float32(np.random.rand(5, 100, 100, 3)) # 随机输入



sess = tf.Session()
images = tf.placeholder("float", [5, 100, 100, 3])
feed_dict = {images : x_data}
y = conv_layer(images)
begin = clock()
for step in xrange(0, 201):
	end = clock()
	usingTime = end - begin
	print("computings times: %d and using time: %f second" % (step, usingTime))
	y_predict = sess.run(y, feed_dict = feed_dict)


