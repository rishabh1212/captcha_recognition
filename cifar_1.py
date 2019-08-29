def init(count):
	import scipy.io as spio
	from skimage.color import rgb2gray
	import pickle
	import matplotlib.pyplot as plt
	import numpy as np
	def dic1():
		with open('cifar-10-batches-py/data_batch_1','rb') as f:
			d = pickle.load(f)
			train_dataset_1 = list((d['data'].reshape(-1,3072).astype(np.float32)-np.array([127.5]*3072))*(1/255.0))
			print(len(train_dataset_1))
			del d
			f.close()
			spio.savemat('cifar_1_train.mat',{'X':np.array(train_dataset_1)})
		with open('cifar-10-batches-py/data_batch_2','rb') as f:
			d = pickle.load(f)
			train_dataset_1 = list((d['data'].reshape(-1,3072).astype(np.float32)-np.array([127.5]*3072))*(1/255.0))
			print(len(train_dataset_1))
			del d
			f.close()
			spio.savemat('cifar_2_train.mat',{'X':np.array(train_dataset_1)})
		with open('cifar-10-batches-py/data_batch_3','rb') as f:
			d = pickle.load(f)
			train_dataset_1 = list((d['data'].reshape(-1,3072).astype(np.float32)-np.array([127.5]*3072))*(1/255.0))
			print(len(train_dataset_1))
			del d
			f.close()
			spio.savemat('cifar_3_train.mat',{'X':np.array(train_dataset_1)})
		with open('cifar-10-batches-py/data_batch_4','rb') as f:
			d = pickle.load(f)
			train_dataset_1 = list((d['data'].reshape(-1,3072).astype(np.float32)-np.array([127.5]*3072))*(1/255.0))
			print(len(train_dataset_1))
			del d
			f.close()
			spio.savemat('cifar_4_train.mat',{'X':np.array(train_dataset_1)})
		with open('cifar-10-batches-py/data_batch_5','rb') as f:
			d = pickle.load(f)
			train_dataset_1 = list((d['data'].reshape(-1,3072).astype(np.float32)-np.array([127.5]*3072))*(1/255.0))
			print(len(train_dataset_1[0]))
			del d
			f.close()	
			spio.savemat('cifar_5_train.mat',{'X':np.array(train_dataset_1)})
		print("char_or_not_train.p")
	def dic2():
		with open('cifar-10-batches-py/test_batch','rb') as f:
			d = pickle.load(f)
			y = list((d['data'].reshape(-1,3072).astype(np.float32)-np.array([127.5]*3072))*(1/255.0))
			del d
			f.close()
		char_or_not_test = {}
		char_or_not_test['X'] = np.array(list(y))
		spio.savemat("cifar_test.mat",char_or_not_test)
		print("char_or_not_test.p")
	def dic3():
		with open("train_32x32.mat") as f:
			i=0
			d = spio.loadmat(f)
			d['X'] = d['X']
			d['y'] = d['y']
			train_dataset = []
			train_labels = []
			j=0
			k=0
			while i<73257:#73257
				train_dataset.append(rgb2gray((d['X'][:,:,:,i])))
				train_labels.append(d['y'][i][0])
				i+=1
				k+=1
				if k==14651:
					k=0
					train_dataset = np.array(train_dataset)
					print(train_dataset.shape)
					f.close()
					print(len(train_labels),max(train_labels),min(train_labels))
					spio.savemat("char_rec_train_g_"+str(j)+".mat",{'X':train_dataset,'Y':train_labels})
					train_dataset=[]
					train_labels=[]
					j+=1
					print("char_rec_train.p")
		# return train_dataset
	def dic4():
		with open("test_32x32.mat") as f:
			d = spio.loadmat(f)
			test_dataset = []
			test_labels = []
			plt.imshow(d['X'][:,:,:,100][0])
			for i in range(26032):
				test_dataset.append(rgb2gray((d['X'][:,:,:,i])))
				test_labels.append(d['y'][i][0])
			del d
			f.close()
		print(test_dataset[0])
		char_rec_test = {}
		char_rec_test['X'] = np.array(test_dataset)
		char_rec_test['Y'] = np.array(test_labels)
		print(len(test_labels))
		#spio.savemat("char_rec_test_g.mat",char_rec_test)
		char_rec_test.clear()
		print("char_rec_test_g.p")
		return test_dataset
	#dic3()
	#dic1()
	dic4()
	# dic2()
x=init(0)