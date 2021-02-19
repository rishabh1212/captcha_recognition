import cv2
import cv
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread
def fff():
	train_data = []
	train_label = []
	with open('allfnt.txt') as f:
		x = f.readlines()
		z = []
		for y in x:
			z.append(y[0:len(y)-1])
		x = 0
		f.close()
	for x in z:
		l = int(x[6]+x[7]+x[8])
		if l>10:
			continue
		y = imread('Fnt/'+x+'.png')
		y = y/255.0
		y = cv2.resize(y, (32, 32), interpolation=cv2.INTER_CUBIC)
		train_data.append(y)
		train_label.append(l)
	with open('allhnd.txt') as f:
		x = f.readlines()
		z = []
		for y in x:
			z.append(y[0:len(y)-1])
		x = 0
		f.close()
	for x in z:
		l = int(x[10]+x[11]+x[12])
		if l>10:
			continue
		y = imread('Hnd/'+x+'.png')
		y = rgb2gray(y)
		y = cv2.resize(y, (32, 32), interpolation=cv2.INTER_CUBIC)
		train_data.append(y)
		train_label.append(l)
	
	with open('all__bad.txt') as f:
		x = f.readlines()
		z = []
		for y in x:
			z.append(y[0:len(y)-1])
		x = 0
		f.close()
	for x in z:
		if 'Msk' in x:
			continue
		l = int(x[18]+x[19]+x[20])
		if l>10:
			continue
		y = imread('Img/'+x+'.png')
		y = rgb2gray(y)
		y = cv2.resize(y, (32, 32), interpolation=cv2.INTER_CUBIC)
		train_data.append(y)
		train_label.append(l)
	with open('all_good.txt') as f:
		x = f.readlines()
		z = []
		for y in x:
			z.append(y[0:len(y)-1])
		x = 0
		f.close()
	for x in z:
		l = int(x[18]+x[19]+x[20])
		if l>10:
			continue
		y = imread('Img/'+x+'.png')
		y = rgb2gray(y)
		y = cv2.resize(y, (32, 32), interpolation=cv2.INTER_CUBIC)
		train_data.append(y)
		train_label.append(l)
	print(len(train_label),len(train_data))
	return train_data
#fff()