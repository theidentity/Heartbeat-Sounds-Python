import os
import pandas as pd
import numpy as np


def get_sum(path):
	count = sum([len(files) for root,dirs,files in os.walk(path)])
	print '***',path,':',count

def get_count(path):

	for root,dirs,files in os.walk(path):
		if len(files)==0:
			get_sum(root)
		else:
			print root,':',len(files)

def analyze_distribution(path):
	df = pd.read_csv(path)
	labels = df['label']

	items,counts = np.unique(labels,return_counts=True)

	print '-----------DISTRI----------'
	for item,count in zip(items,counts):
		print item,':',count




get_count('data/image_folds/fold1/')

# analyze_distribution('data/csvs/all.csv')
# analyze_distribution('data/csvs/aug_all.csv')