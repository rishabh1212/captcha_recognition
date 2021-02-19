import scipy.io as spio
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pickle
import numpy
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
def dic(i):
	with open('cifar-10-batches-py/data_batch_'+str(i+1),'rb') as f:
		d = pickle.load(f)
		train_dataset = []
		for x in d['data']:
			train_dataset.append((rgb2gray(x.reshape(-1,32,32,3)))[0])
		#train_dataset = list(d['X'])
		# print(train_dataset[0])
		# plt.imshow(train_dataset[4])
		# plt.show()
		train_label = list([0]*(10000))
		del d
		f.close()
	with open('char_rec_train_g_'+str(i)+'.mat') as f:
		d = spio.loadmat(f)
		train_dataset += list(d['X'])
		train_label += list([1]*(len(train_dataset)-len(train_label)))
		del d
		f.close()
	a,b = shuffle_in_unison(numpy.array(train_dataset), numpy.array(train_label))
	train_dataset = 0
	train_label = 0
	#spio.savemat("char_or_not_train_"+str(i)+".mat",{'X':a,'Y':b})
	print(len(a))
	print(len(b))
	return 1
for i in range(5):
	k = dic(i)
with open('cifar-10-batches-py/test_batch','rb') as f:
	d = pickle.load(f)
	test_dataset = []
	for x in d['data']:
		test_dataset.append((rgb2gray(x.reshape(-1,32,32,3)))[0])
	print(test_dataset[0])
	test_label = list([0]*(len(test_dataset)))
	del d
	f.close()
print(4)
with open('char_rec_test_g.mat') as f:
	d = spio.loadmat(f)
	test_dataset += list(d['X'])
	test_label += list([1]*(len(test_dataset)-len(test_label)))
	del d
	#spio.savemat("char_or_not_test.mat",{'X':test_dataset,'Y':test_label})
	f.close()
print(5)