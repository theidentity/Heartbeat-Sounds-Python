from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from multiprocessing import Pool,cpu_count
import librosa
import numpy as np
import librosa.display
import pandas as pd
import os


def create_folder(path):
	if not os.path.exists(path):
		print path
		os.makedirs(path)

def get_quadmesh(wav_path,save_path):
	
	name = wav_path.split('/')[-1].split('.')[0]
	print name

	sample_rate, samples = wavfile.read(wav_path)
	frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

	plt.pcolormesh(times, frequencies, spectrogram)
	plt.imshow(spectrogram)

	plt.savefig(save_path+name+'.jpg')

def get_mel_spectorgram(wav_path,save_path):
	
	name = wav_path.split('/')[-1].split('.')[0]
	print name

	y,sr = librosa.load(wav_path)
	S = librosa.feature.melspectrogram(y,sr,n_mels=128)
	log_S = librosa.amplitude_to_db(S,ref=np.max)
	plt.figure(figsize=(8,8))

	librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
	# plt.colorbar(format='%+02.0f dB')
	# plt.tight_layout()
	plt.savefig(save_path+name+'.jpg')

def process_wav(wav_path):
	create_folder('data/spectrograms/quad_mesh/')
	create_folder('data/spectrograms/mel/')
	# get_quadmesh(wav_path,save_path='data/spectrograms/quad_mesh/')
	get_mel_spectorgram(wav_path,save_path='data/spectrograms/mel/')





if __name__ == '__main__':
	
	files = pd.read_csv('data/csvs/all.csv')['fname']
	wav_paths = [''.join(['data/sounds/',x]) for x in files]

	pool = Pool(cpu_count())
	pool.map(process_wav,wav_paths[:])