import pandas as pd
import numpy as np
from glob import glob
import os

def get_labelled(path):
	df = pd.read_csv(path,sep=',')
	df = df[['fname','label']]
	
	indices = df['label'].isnull()
	df = df[~indices]

	labels = df['label']
	labels = [x.split('/')[-1] for x in labels]
	df['label'] = labels
	
	return df

def clean_labels(df):

	df['label'][df['label']=='extrahls'] = 'extrastole'

	labels = df['label']
	labels = [x.replace('__','_') for x in labels]
	df['label'] = labels
	return df

def remove_artifact_label(df):
	indices = df['label'] == 'artifact'
	df = df[~indices]
	return df 

def print_counts(df):
	items,counts = np.unique(df['label'],return_counts=True)
	for item,count in zip(items,counts):
		print item,':',count

def fix_paths(df):
	paths = df['fname']
	paths = [x.split('/')[-1] for x in paths]
	paths = [x.replace('__','_') for x in paths]
	paths = [x.replace('extrastole','extrahls') for x in paths]
	df['fname'] = paths
	return df

def rename_wavs(path):
	wavs = glob(path)

	for wav in wavs:
		new_name = wav.replace('__','_')
		new_name = wav.replace('extrastole','extrahls')
		os.rename(wav,new_name)

def remove_prefixes(df):
	names = df['fname']
	names = [x.replace('Btraining_','') for x in names]
	names = [x.replace('Btraining_normal_Btraining_','') for x in names]
	names = [x.replace('extrastole','extrahls') for x in names]
	df['fname'] = names
	return df

if __name__ == '__main__':
	
	df_A = get_labelled('data/csvs/set_a.csv')
	df_B = get_labelled('data/csvs/set_b.csv')

	df = pd.concat([df_A,df_B])

	df = clean_labels(df)
	# df = remove_artifact_label(df)
	df = fix_paths(df)
	df = remove_prefixes(df)

	print_counts(df)

	df.to_csv('data/csvs/all.csv',index=False)

	rename_wavs('data/sounds/*wav')