def init():
  import numpy as np
  import tensorflow as tf
  from six.moves import cPickle as pickle
  from six.moves import range
  import copy
  image_size = 28
  num_labels = 10
  with open('notMNIST.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    train_dataset = train_dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    print(train_dataset[1])
    valid_dataset = valid_dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    test_dataset = test_dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
    valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
    test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
  with open('mnist.pkl', 'rb') as f:
    save = pickle.load(f)
    train_dataset_1 = save[0][0]
    train_labels_1 = save[0][1]
    valid_dataset_1 = save[1][0]
    valid_labels_1 = save[1][1]
    test_dataset_1 = save[2][0]
    test_labels_1 = save[2][1]
    del save  # hint to help gc free up memory
    train_dataset_1 = train_dataset_1.reshape((-1, image_size * image_size)).astype(np.float32)
    valid_dataset_1 = valid_dataset_1.reshape((-1, image_size * image_size)).astype(np.float32)
    test_dataset_1 = test_dataset_1.reshape((-1, image_size * image_size)).astype(np.float32)
    train_labels_1 = (np.arange(num_labels) == train_labels_1[:,None]).astype(np.float32)
    valid_labels_1 = (np.arange(num_labels) == valid_labels_1[:,None]).astype(np.float32)
    test_labels_1 = (np.arange(num_labels) == test_labels_1[:,None]).astype(np.float32)
    print('Training set', train_dataset_1.shape, train_labels_1.shape)
    print('Validation set', valid_dataset_1.shape, valid_labels_1.shape)
    print('Test set', test_dataset_1.shape, test_labels_1.shape)
  train_dataset = list(train_dataset)
  train_labels = list(train_labels)
  valid_dataset = list(valid_dataset)
  valid_labels = list(valid_labels)
  test_dataset = list(test_dataset)
  test_labels = list(test_labels)
  for x in train_dataset_1:
    train_dataset.append(x)
  for x in train_labels_1:
    train_labels.append(x)
  for x in valid_dataset_1:
    valid_dataset.append(x)
  for x in valid_labels_1:
    valid_labels.append(x)
  for x in test_dataset_1:
    test_dataset.append(x)
  for x in test_labels_1:
    test_labels.append(x)
  # train_dataset = np.array(train_dataset)
  # train_labels = np.array(train_labels)
  # valid_dataset = np.array(valid_dataset)
  # valid_labels = np.array(valid_labels)
  # test_dataset = np.array(test_dataset)
  # test_labels = np.array(test_labels)
  # print('Training set', train_dataset.shape, train_labels.shape)
  # print('Validation set', valid_dataset.shape, valid_labels.shape)
  # print('Test set', test_dataset.shape, test_labels.shape)
  from sklearn.decomposition import RandomizedPCA
  f = open('cifar-10-batches-py/data_batch_1','rb')
  d_1 = pickle.load(f)
  data_10 = list((copy.deepcopy(d_1['data'])))
  label_10 = list(copy.deepcopy(d_1['labels']))
  f = open('cifar-10-batches-py/data_batch_2','rb')
  d_2 = pickle.load(f)
  data_10 = data_10 + list((copy.deepcopy(d_2['data'])))
  label_10 = label_10 + list(copy.deepcopy(d_2['labels']))
  f = open('cifar-10-batches-py/data_batch_3','rb')
  d_3 = pickle.load(f)
  data_10 += list((copy.deepcopy(d_3['data'])))
  label_10 += list(copy.deepcopy(d_3['labels']))
  f = open('cifar-10-batches-py/data_batch_4','rb')
  d_4 = pickle.load(f)
  data_10 += list((copy.deepcopy(d_4['data'])))
  label_10 += list(copy.deepcopy(d_4['labels']))
  f = open('cifar-10-batches-py/data_batch_5','rb')
  d_5 = pickle.load(f)
  data_10 += list((copy.deepcopy(d_5['data'])))
  label_10 += list(copy.deepcopy(d_5['labels']))
  data_1 = np.array(data_10)                                                      
  data_10 = list(RandomizedPCA(784).fit(data_1).transform(data_1))
  f = open('cifar-10-batches-py/test_batch','rb')
  d_t = pickle.load(f)
  t_data_10 = list(RandomizedPCA(784).fit(copy.deepcopy(d_t['data'])).transform(copy.deepcopy(d_t['data'])))
  t_label_10 = list(copy.deepcopy(d_t['labels']))
  f.close()
  char_id_train = np.array(data_10[0:250000] + train_dataset[0:250000])
  char_label_train =np.array([0]*(100) + [1]*(100))
  char_id_test = np.array(t_data_10 + test_dataset)
  char_label_test = np.array([0]*(len(t_data_10)) + [1]*len(test_dataset))
  n_components = 300
  pca = RandomizedPCA(n_components=n_components, whiten=True).fit(char_id_train)
  char_id_train_pca = pca.transform(char_id_train)
  char_id_test_pca = pca.transform(char_id_test)
  print("returning data_set...................")
  return char_id_train_pca,char_label_train,char_id_test_pca,char_label_test
x = init()
