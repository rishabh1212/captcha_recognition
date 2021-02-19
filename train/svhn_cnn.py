import numpy as np 
import scipy.io as spio
import tensorflow as tf 
# import stage_1_cnn_gray
import image_part
import sys
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, 32, 32, 3)).astype(np.float32)
	z = []
 	for x in labels:
 		if x==1:
 			z.append(np.array([1,0,0,0,0,0,0,0,0,0]).astype(np.float32))
 		if x==2:
 			z.append(np.array([0,1,0,0,0,0,0,0,0,0]).astype(np.float32))
 		if x==3:
 			z.append(np.array([0,0,1,0,0,0,0,0,0,0]).astype(np.float32))
 		if x==4:
 			z.append(np.array([0,0,0,1,0,0,0,0,0,0]).astype(np.float32))
 		if x==5:
 			z.append(np.array([0,0,0,0,1,0,0,0,0,0]).astype(np.float32))
 		if x==6:
 			z.append(np.array([0,0,0,0,0,1,0,0,0,0]).astype(np.float32))
 		if x==7:
 			z.append(np.array([0,0,0,0,0,0,1,0,0,0]).astype(np.float32))
 		if x==8:
 			z.append(np.array([0,0,0,0,0,0,0,1,0,0]).astype(np.float32))
 		if x==9:
 			z.append(np.array([0,0,0,0,0,0,0,0,1,0]).astype(np.float32))
 		if x==10:
 			z.append(np.array([0,0,0,0,0,0,0,0,0,1]).astype(np.float32))
  	return dataset, np.array(z)
with open('char_rec_test.mat') as f:
	d = spio.loadmat(f)
	test_dataset = d['X']
	test_label = d['Y'][0]
	test_dataset, test_label = reformat(np.array(test_dataset), np.array(test_label))
	del d
	f.close()
#test_image = sys.argv[1]
#test_dataset = xxx.ggg(image_part,test_image)
with open('char_rec_train_0.mat') as f:
	d = spio.loadmat(f)
	train = list(d['X'])
	label = list(d['Y'][0])
	print(1)
	f.close()
with open('char_rec_train_1.mat') as f:
	d = spio.loadmat(f)
	train += list(d['X'])
	label += list(d['Y'][0])
	print(1)
	f.close()
with open('char_rec_train_2.mat') as f:
	d = spio.loadmat(f)
	train += list(d['X'])
	label += list(d['Y'][0])
	print(1)
	f.close()
with open('char_rec_train_3.mat') as f:
	d = spio.loadmat(f)
	train += list(d['X'])
	label += list(d['Y'][0])
	print(1)
	f.close()
with open('char_rec_train_4.mat') as f:
	d = spio.loadmat(f)
	train += list(d['X'])
	label += list(d['Y'][0])
	print(1)
	f.close()
train = np.array(train)
label = np.array(label)
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
	saver = tf.train.Saver()
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_label, logits = logits))
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
	train_pred = tf.nn.softmax(logits)
	test_pred = tf.nn.softmax(cnn(tf_test_dataset))
print("session------------")
with tf.Session(graph = fg) as session:
	tf.global_variables_initializer().run()
	for x in range(15001):
		offset = (x*128)%(14650*5-128)	
		batch_data, batch_label = reformat(train[offset:offset+128], label[offset:offset+128])
		feed = {tf_train_dataset : batch_data, tf_train_label : batch_label}
		_, l, pred = session.run([optimizer, loss, train_pred], feed_dict = feed)
		if(x%50==0):
			print('Minibatch loss at step %d: %f' % (x, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(pred, batch_label))
	save_path = saver.save(session, 'svhn_cnn.ckpt')
	train = 0
	label = 0
	print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_label))