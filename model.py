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
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
	z = []
 	for x in labels:
 		if x==1:
 			z.append(np.array([0,1]).astype(np.float32))
 		else:
 			z.append(np.array([1,0]).astype(np.float32))
  	return dataset, np.array(z)
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
def ggg(image_part,test_image):
	fg = tf.Graph()
	batch_size = 128
	print(1)
	sss,test_dataset = image_part.fff(test_image)
	ff = copy.deepcopy(sss)
	print(len(test_dataset))
	test_label = [0]*len(test_dataset)
	test_dataset, test_label = reformat(np.array(test_dataset), np.array(test_label))
	import random
	with fg.as_default():
		tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, 32, 32,1))
		tf_train_label = tf.placeholder(tf.float32, shape = (batch_size, 2))
		tf_test_dataset = tf.constant(test_dataset, dtype = tf.float32)
		w_11 = tf.Variable(tf.truncated_normal([4, 4, 1, 2], stddev = 0.1),name = 'w_11')
		b_11 = tf.Variable(tf.zeros([2]),name = 'b_11')
		# w_2 = tf.Variable(tf.truncated_normal([5, 5, 5, 10], stddev = 0.1))
		# b_2 = tf.Variable(tf.constant(1.0, shape = [10]))
		w_31 = tf.Variable(tf.truncated_normal([512, 4], stddev = 0.1),name = 'w_31')
		b_31 = tf.Variable(tf.constant(1.0, shape = [4]),name = 'b_31')
		w_41 = tf.Variable(tf.truncated_normal([4, 2], stddev = 0.1), name = 'w_41')
		b_41 = tf.Variable(tf.constant(1.0, shape = [2]), name = 'b_41')
		def cnn(data):
			conv = tf.nn.conv2d(data, w_11, [1, 2, 2, 1], padding='SAME')
			hidden = tf.nn.relu(conv + b_11)
			# conv = tf.nn.conv2d(hidden, w_2, [1, 2, 2, 1], padding='SAME')
			# hidden = tf.nn.relu(conv + b_2)
			shape = hidden.get_shape().as_list()
			reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
			hidden = tf.nn.relu(tf.matmul(reshape, w_31) + b_31)
			return tf.matmul(hidden, w_41) + b_41
		new_saver = tf.train.Saver()
		logits = cnn(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_label, logits = logits))
		optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
		train_pred = tf.nn.softmax(logits)
		test_pred = tf.nn.softmax(cnn(tf_test_dataset))
	print("session------------")
	from matplotlib import pyplot as plt
	with tf.Session(graph = fg) as session:
		tf.global_variables_initializer().run()
		new_saver = tf.train.import_meta_graph('stage_1_cnn.ckpt.meta')
		new_saver.restore(session, tf.train.latest_checkpoint('./'))
		x = test_pred.eval()
		train = 0
		label = 0	
		img = []														
		for i in range(len(ff)):
			if x[i][0]<x[i][1]:
				plt.imshow(ff[i])
				img.append(ff[i])
				plt.show()
				print((np.exp(x[i][1])/(np.exp(x[i][1])+np.exp(x[i][0]))))
	return img
img = ggg(image_part,sys.argv[1])
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