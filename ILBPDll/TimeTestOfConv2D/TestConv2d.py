import numpy as np
import tensorflow as tf
import time

input_data = tf.Variable( np.random.rand(1,2000,2000,1), dtype = np.float32 )
filter_data = tf.Variable( np.random.rand(8, 8, 1, 8), dtype = np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides = [1, 1, 1, 1], padding = 'SAME')

with tf.Session() as sess:
    #result = sess.run(op)#run operation
    sess.run(tf.global_variables_initializer())

    for i in range(0,100):
        start = time.perf_counter()
        sess.run(y)

        end = time.perf_counter()
        elapsed = end - start
        print("elapsed time = {:.12f} seconds".format(elapsed))

    