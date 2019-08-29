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
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
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
	train_dataset = []
	for i in range(5):
		with open('cifar-10-batches-py/data_batch_'+str(i+1),'rb') as f:
				d = pickle.load(f)
				for x in d['data']:
					train_dataset.append(rgb2gray(x.reshape(-1,32,32,3))[0])
				train_label = list([0]*(10000))
				del d
				f.close()
	import random
	random.shuffle(train_dataset)
	r = char_74k.fff()
	train_dataset = train_dataset[0:int(len(r))]
	random.shuffle(r)
	train_dataset = list(train_dataset)
	r = list(r)
	i=0
	j=0
	train = []
	label = []
	while i < len(r):
		train += train_dataset[min(j+64,len(train_dataset)-1)-64:min(j+64,len(train_dataset)-1)]
		print(len(train_dataset[min(j+64,len(train_dataset)-1)-64:min(j+64,len(train_dataset)-1)]))
		train += r[min(i+64,len(r)-1)-64:min(i+64,len(r)-1)]
		label += [0]*64+[1]*64
		print(len(label))
		i=i+64
		j=j+64
	print(len(train))
	ll = len(label)
	train = np.array(train)
	label = np.array(label)
	train , label = shuffle_in_unison(train,label)
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
		saver = tf.train.Saver()
		logits = cnn(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_label, logits = logits))
		optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
		train_pred = tf.nn.softmax(logits)
		test_pred = tf.nn.softmax(cnn(tf_test_dataset))
	print("session------------")
	from matplotlib import pyplot as plt
	with tf.Session(graph = fg) as session:
		tf.global_variables_initializer().run()
		for x in range(7001):
			offset = (x*batch_size)%(ll-batch_size)	
			batch_data, batch_label = reformat(train[offset:offset+batch_size], label[offset:offset+batch_size])
			feed = {tf_train_dataset : batch_data, tf_train_label : batch_label}
			_, l, pred = session.run([optimizer, loss, train_pred], feed_dict = feed)
			if(x%50==0):
				print('Minibatch loss at step %d: %f' % (x, l))
				print('Minibatch accuracy: %.1f%%' % accuracy(pred, batch_label))
		x = test_pred.eval()
		save_path = saver.save(session, 'stage_1_cnn.ckpt')
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
ggg(image_part,'../xxx/20.jpg')