import numpy as np
import pandas as pd

import librosa
from random import getrandbits
import sys, getopt, os

from multiprocessing import Pool,cpu_count
import shutil


def random_onoff():				
	# randomly turns on or off
	return bool(getrandbits(1))

# returns a list of augmented audio data, stereo or mono
def augment_data(y,sr,n_augment=0,allow_speedandpitch=True,allow_pitch=True,allow_speed=True,
	allow_dyn=True,allow_noise = True,allow_timeshift=True,tab=""):

	np.random.seed(42)
	mods = [y]				  # always returns the original as element zero
	length = y.shape[0]

	for i in range(n_augment):
		# print(tab+"augment_data: ",i+1,"of",n_augment)
		y_mod = y
		count_changes = 0

		# change speed and pitch together
		if (allow_speedandpitch) and random_onoff():   
			length_change = np.random.uniform(low=0.9,high=1.1)
			speed_fac = 1.0  / length_change
			# print(tab+"	resample length_change = ",length_change)
			tmp = np.interp(np.arange(0,len(y),speed_fac),np.arange(0,len(y)),y)
			#tmp = resample(y,int(length*lengt_fac))	# signal.resample is too slow
			minlen = min( y.shape[0], tmp.shape[0])	 # keep same length as original; 
			y_mod *= 0									# pad with zeros 
			y_mod[0:minlen] = tmp[0:minlen]
			count_changes += 1

		# change pitch (w/o speed)
		if (allow_pitch) and random_onoff():   
			bins_per_octave = 24		# pitch increments are quarter-steps
			pitch_pm = 4								# +/- this many quarter steps
			pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)   
			# print(tab+"	pitch_change = ",pitch_change)
			y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
			count_changes += 1

		# change speed (w/o pitch), 
		if (allow_speed) and random_onoff():   
			speed_change = np.random.uniform(low=0.9,high=1.1)
			# print(tab+"	speed_change = ",speed_change)
			tmp = librosa.effects.time_stretch(y_mod, speed_change)
			minlen = min( y.shape[0], tmp.shape[0])		# keep same length as original; 
			y_mod *= 0									# pad with zeros 
			y_mod[0:minlen] = tmp[0:minlen]
			count_changes += 1

		# change dynamic range
		if (allow_dyn) and random_onoff():  
			dyn_change = np.random.uniform(low=0.5,high=1.1)  # change amplitude
			# print(tab+"	dyn_change = ",dyn_change)
			y_mod = y_mod * dyn_change
			count_changes += 1

		# add noise
		if (allow_noise) and random_onoff():  
			noise_amp = 0.005*np.random.uniform()*np.amax(y)  
			if random_onoff():
				# print(tab+"	gaussian noise_amp = ",noise_amp)
				y_mod +=  noise_amp * np.random.normal(size=length)  
			else:
				# print(tab+"	uniform noise_amp = ",noise_amp)
				y_mod +=  noise_amp * np.random.normal(size=length)  
			count_changes += 1

		# shift in time forwards or backwards
		if (allow_timeshift) and random_onoff():
			timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
			# print(tab+"	timeshift_fac = ",timeshift_fac)
			start = int(length * timeshift_fac)
			if (start > 0):
				y_mod = np.pad(y_mod,(start,0),mode='constant')[0:y_mod.shape[0]]
			else:
				y_mod = np.pad(y_mod,(0,-start),mode='constant')[0:y_mod.shape[0]]
			count_changes += 1

		# last-ditch effort to make sure we made a change (recursive/sloppy, but...works)
		if (0 == count_changes):
			# print("No changes made to signal, trying again")
			mods.append(augment_data(y,sr,n_augment = 1, tab="	  ")[1] )
		else:
			mods.append(y_mod)

	return mods


def augment_wav(wav_path,label,n_augment):

	# wav_path,label = args
	# print wav_path,label
	print label
	in_path = 'data/sounds/'+wav_path

	y,sr = librosa.load(in_path)

	mods = augment_data(y, sr, n_augment=n_augment)
	librosa.output.write_wav('data/augmented/'+wav_path,y,sr)

	paths = []
	paths.append(wav_path)

	for i in range(len(mods)-1):
		out_path = "data/augmented/aug_"+str(i+1)+'_'+wav_path
		paths.append('aug_'+str(i+1)+'_'+wav_path)
		librosa.output.write_wav(out_path,mods[i+1],sr)

	return paths,[label for i in range((len(paths)))]

def augment_wav_wrapper(args):
	paths,labels = augment_wav(*args)
	return (paths,labels)

def transfer_normal(df):
	paths = df['fname']
	in_paths = ['data/sounds/'+x for x in paths]
	out_paths = ['data/augmented/'+x for x in paths]

	for src,dst in zip(in_paths,out_paths):
		shutil.copy(src,dst)

def augment_rest(df):
	wav_paths = df['fname']
	labels = df['label']

	pool = Pool(cpu_count())
	results = pool.map(augment_wav_wrapper,zip(wav_paths,labels,[10 for i in range(len(wav_paths))]))
	
	aug_paths = []
	aug_labels = []
	
	for result in results:
		(aug_path,aug_label) = result
		aug_paths += aug_path	
		aug_labels += aug_label	


	df = pd.DataFrame()
	df['fname'] = aug_paths
	df['label'] = aug_labels
	return df


if __name__ == '__main__':
	df = pd.read_csv('data/csvs/all.csv')

	# augment ones without enough samples
	indices = df['label'] == 'normal'
	df_A = df[indices]
	df_B = df[~indices]

	transfer_normal(df_A)
	df_B = augment_rest(df_B)

	df = pd.concat([df_A,df_B])
	df.to_csv('data/csvs/aug_all.csv',index=False)
