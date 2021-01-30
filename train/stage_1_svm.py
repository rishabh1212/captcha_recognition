import numpy as np 
import scipy.io as spio
import copy
import matplotlib.pyplot as plt
import image_part 
from skimage.color import rgb2gray
import pickle
import char_74k
import cv2
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
	z = []
 	for x in labels:
 		if x==1:
 			z.append(np.array([0,1]).astype(np.float32))
 		else:
 			z.append(np.array([1,0]).astype(np.float32))
  	return dataset, np.array(z)
def ggg(image_part,test_data):
	sss,test_dataset = image_part.fff(test_data)
	ff = copy.deepcopy(sss)
	print(len(test_dataset))
	test_label = [0]*len(test_dataset)
	test_dataset, test_label = reformat(np.array(test_dataset), np.array(test_label))
	train_dataset = []
	for i in range(5):
		with open('cifar-10-batches-py/data_batch_'+str(i+1),'rb') as f:
				d = pickle.load(f)
				for x in d['data']:
					train_dataset.append((rgb2gray(x.reshape(-1,32,32,3)))[0])
				train_label = list([0]*(10000))
				del d
				f.close()
	import random
	random.shuffle(train_dataset)
	r = char_74k.fff()
	train_dataset = train_dataset[0:len(r)]
	random.shuffle(r)
	train_dataset = list(train_dataset)
	r = list(r)
	i=0
	train = []
	label = []
	while i < len(r):
		train += train_dataset[min(i+64,len(train_dataset)-1)-64:min(i+64,len(train_dataset)-1)]
		print(len(train_dataset[min(i+64,len(train_dataset)-1)-64:min(i+64,len(train_dataset)-1)]))
		train += r[min(i+64,len(r)-1)-64:min(i+64,len(r)-1)]
		label += [0]*64+[1]*64
		print(len(label))
		i=i+64
	print(len(train))
	ll = len(label)
	train = np.array(train)
	label = np.array(label)
	from sklearn import svm
	clf = svm.SVC()
	print(1)
	clf.fit((train.reshape(-1,1024)),label)
	print(2)
	x = clf.predict((test_dataset.reshape(-1,1024)))
	img = []
	for a in range(len(x)):
		if x[a]==1:
			img.append(ff[a])
			plt.imshow(ff[a])
			plt.show()
	return img
ggg(image_part,'../xxx/20.jpg')