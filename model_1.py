import numpy as np
import scipy.io as spio
import tensorflow as tf
import copy
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pickle
import char_74k
import cv2
import image_part 
import sys
import scipy.io as spio
with open('testimg.mat') as f:
	d = spio.loadmat(f)
	img = d['X'] 
	del d
test_dataset = np.array(img)
batch_size = 128
fg = tf.Graph()
with fg.as_default():
	tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, 32, 32, 3))
	tf_train_label = tf.placeholder(tf.float32, shape = (batch_size, 10))
	tf_test_dataset = tf.constant(test_dataset, dtype = tf.float32)
	w_1 = tf.Variable(tf.truncated_normal([5, 5, 3, 10], stddev = 0.1),name = 'w_1')
	b_1 = tf.Variable(tf.zeros([10]),name = 'b_1')
	w_2 = tf.Variable(tf.truncated_normal([5, 5, 10, 15], stddev = 0.1),name = 'w_2')
	b_2 = tf.Variable(tf.constant(1.0, shape = [15]),name = 'b_2')
	w_25 = tf.Variable(tf.truncated_normal([5, 5, 15, 80], stddev = 0.1),name = 'w_25')
	b_25 = tf.Variable(tf.constant(1.0, shape = [80]),name = 'b_25')
	w_3 = tf.Variable(tf.truncated_normal([32//4*32//16*5, 128], stddev = 0.1),name = 'w_3')
	b_3 = tf.Variable(tf.constant(1.0, shape = [128]),name = 'b_3')
	w_4 = tf.Variable(tf.truncated_normal([128, 10], stddev = 0.1),name = 'w_4')
	b_4 = tf.Variable(tf.constant(1.0, shape = [10]),name = 'b_4')
	def cnn(data):
		conv = tf.nn.conv2d(data, w_1, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b_1)
		hidden = tf.nn.avg_pool(hidden, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
		conv = tf.nn.conv2d(hidden, w_2, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b_2)
		hidden = tf.nn.avg_pool(hidden, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
		conv = tf.nn.conv2d(hidden, w_25, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + b_25)
		hidden = tf.nn.avg_pool(hidden, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, w_3) + b_3)
		return tf.matmul(hidden, w_4) + b_4
	logits = cnn(tf_train_dataset)
	new_saver = tf.train.Saver()
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_label, logits = logits))
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
	train_pred = tf.nn.softmax(logits)
	test_pred = tf.nn.softmax(cnn(tf_test_dataset))
print("session------------")
with tf.Session(graph = fg) as session:
	tf.global_variables_initializer().run()
	new_saver = tf.train.import_meta_graph('svhn_cnn.ckpt.meta')
	new_saver.restore(session, tf.train.latest_checkpoint('./'))
	print(test_pred.eval())
	#print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_label))