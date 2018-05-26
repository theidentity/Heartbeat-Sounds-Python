import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,StratifiedKFold
import os
import shutil


def create_folder(path):
	if not os.path.exists(path):
		print path
		os.makedirs(path)

def create_folder_skeleton(n_splits):
	base_folder = 'data/image_folds/'
	labels = ['extrastole','murmur','normal']
	for i in range(n_splits):
		fold_name = 'fold'+str(i+1)+'/'
		for label in labels:
			folder_path = base_folder+fold_name+'train/'+label+'/'
			create_folder(folder_path)
			folder_path = base_folder+fold_name+'validation/'+label+'/'
			create_folder(folder_path)

def get_data():
	df = pd.read_csv('data/csvs/all.csv')
	X = df['fname']
	y = df['label']
	return X,y

def split_into_folds(X,y,n_splits=4):

	skf = StratifiedKFold(n_splits=n_splits)
	for train_idx,test_idx in skf.split(X,y):
		train_X = X[train_idx]
		train_y = y[train_idx]
		test_X = X[test_idx]
		test_y = y[test_idx]

		yield (train_X,train_y),(test_X,test_y)

def print_counts(arr):
	items,counts = np.unique(arr,return_counts=True)
	for item,count in zip(items,counts):
		print item,':',count

def copy_to_folder(X,y,fold,train_set=True):
	if train_set:
		base_path = ''.join(['data/image_folds/fold',str(fold),'/train/'])
	else:
		base_path = ''.join(['data/image_folds/fold',str(fold),'/validation/'])

	names = [name.replace('.wav','.jpg') for name in X]
	in_paths = ['data/spectrograms/mel/'+name for name in names]
	out_paths = [base_path+label+'/'+name for name,label in zip(names,y)]

	for src,dst in zip(in_paths,out_paths):
		print src,dst
		shutil.copy(src,dst)

def organize_spectrograms_to_folds(folds,n_splits=4):

	for i in range(n_splits):
		(train_X,train_y),(test_X,test_y) = folds.next()
		copy_to_folder(train_X,train_y,i+1,train_set=True)
		# print_counts(train_y)
		# print_counts(test_y)



def create_placeholder(base_path):
	 for root,dirs,files in os.walk(base_path):
	 	if not dirs:
	 		file = open(root+'/.gitignore','w+')
	 		file.write('')
	 		print root

create_folder_skeleton(n_splits=4)
create_placeholder(base_path='data/')
X,y = get_data()
folds = split_into_folds(X,y,n_splits=4)
organize_spectrograms_to_folds(folds,n_splits=4)