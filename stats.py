import os


def get_count(path):

	for root,dirs,files in os.walk(path):
		print root,':',len(files)


get_count('data/image_folds/fold1/')