
import os
from IPython import embed
train_file = open('airplane.train', 'a+')
trainhalf_file = open('airplane.half', 'a+')

datadir = 'images/train/'
datas = os.listdir(datadir)
datas.sort()
for ddir in datas:
	imgs = os.listdir(datadir+ddir+'/img1/')
	leng = len(imgs)
	imgs.sort()
	for i, img in enumerate(imgs):
		if i <= leng//2:
			trainhalf_file.write('airplane/'+datadir+ddir+'/img1/'+'{}\n'.format(img))
		train_file.write('airplane/'+datadir+ddir+'/img1/'+'{}\n'.format(img))
train_file.close()
trainhalf_file.close()

