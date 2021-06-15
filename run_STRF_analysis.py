#Generalizable EEG encoding models for audiovisual stimuli code 
#Desai et al. 2021 
# Manuscript code 

# import the libraries we need

import scipy.io 
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
from mne import io
import numpy as np
from numpy.polynomial.polynomial import polyfit
from audio_tools import spectools, fbtools, phn_tools #use custom functions for linguistic/acoustic alignment
from scipy.io import wavfile
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import os
import re
import pingouin as pg #stats package 
import pandas as pd
import traceback
import textgrid as tg

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib_venn import venn3, venn3_circles, venn2
from scipy.stats import wilcoxon

from ridge.utils import make_delayed, counter, save_table_file
from ridge.ridge import ridge_corr, bootstrap_ridge, bootstrap_ridge_shuffle, eigridge_corr

import random
import itertools as itools
np.random.seed(0)
random.seed(0)

from scipy import stats
import scipy.optimize

import logging
import math
import glob



def load_ICA_data(subject, data_dir, file='post_ICA', preload=True):
	"""
	Load ICA EEG data (completely processed .fif file)
	Used to call specific subject for running STRFs and plotting correlations

	Parameters
	----------
	subject : string 
		subject ID (i.e. MT0002)
	file : string
		(default: post_ICA)
	preload : bool
		(default : True)	

	Returns
	-------
	ds : mne Format 
		loads EEG data with relevant content saved from the MNE format (see online documentation)
	"""
	DS_dir = 'downsampled_128'
	#datadir='/Users/md42732/Desktop/data/EEG/MovieTrailers/Participants/%s/%s'%(subject, DS_dir)
	if file == 'post_ICA':
		ds = io.read_raw_fif('%s/%s/%s/%s_postICA_rejected.fif'%(data_dir, subject, DS_dir, subject), preload=True)
	else:
		print('No other file previously saved to load')

	return ds


def loadEEGh5(subject, stimulus_class, data_dir,
	eeg_epochs=True, resp_mean = True, binarymat=False, binaryfeatmat = True, envelope=True, pitch=True, gabor_pc10=False, 
	spectrogram=True, binned_pitches=True, spectrogram_scaled=True, scene_cut=True):
	"""
	Load contents saved per subject from large .h5 created, which contains EEG epochs based on stimulus type 
	and corresponding speech features. 
	
	Parameters
	----------
	subject : string 
		subject ID (i.e. MT0002)
	stimulus_class : string 
		MovieTrailers or TIMIT 
	data_dir : string 
		-change this to match where .h5 is along with EEG data 
	eeg_epochs : bool
		determines whether or not to load EEG epochs per stimulus type per participant
		(default : True)
	resp_mean : bool
		takes the mean across epochs for stimuli played more than once 
		(default : True)
	binarymat : bool
		determines whether or not to load 52 unique individual phoneme types 
		(deafult : False)
	binaryfeatmat : bool
		determines whether or not to load 14 unique phonological features 
		(default : True)
	envelope : bool
		determines whether or not to load the acoustic envelope of each stimulus type 
		(default : True)
	pitch : bool
		determines whether or not to load the pitch of each stimulus type 
	binned_pitches: bool
		load pitch which are binned base on frequency 
	gabor_pc10 : bool
		inclusion of visual feature from the gabor wavelets  
		(default : False)
	spectrogram : bool
		load the spectrogram of a sound 
		(default : False)
	scene_cut : bool 
		load visual data from Praat textgrid with transcribed scene cuts 
		(default : False )

	Returns
	-------
	stim_dict : dict
		generates all features for the desired stimulus_class for a given subject as a array within the dict
		the shape of all features are resampled to the shape of phnfeat (phonological features)

	resp_dict : dict
		generates all epochs of features for the desired stimulus_class for a given subject as a array within the dict
		the shape of all epochs are resampled to the shape of phnfeat (phonological features)
	"""	 

	stim_dict = dict()
	resp_dict = dict()
	with h5py.File('%s/fullEEGmatrix.hf5'%(data_dir),'r') as fh:
		print(stimulus_class)
		all_stim = [k for k in fh['/%s' %(stimulus_class)].keys()]
		print(all_stim)
			
		for idx, wav_name in enumerate(all_stim): 
			print(wav_name)
			stim_dict[wav_name] = []
			resp_dict[wav_name] = []
			try:
				epochs_data = fh['/%s/%s/resp/%s/epochs' %(stimulus_class, wav_name, subject)][:]
				phnfeatmat = fh['/%s/%s/stim/phn_feat_timings' %(stimulus_class, wav_name)][:]
				ntimes = phnfeatmat.shape[1] #always resample to the size of phnfeat 
				if binarymat:
					phnmat = fh['/%s/%s/stim/phn_timings' %(stimulus_class, wav_name)][:] 
					stim_dict[wav_name].append(phnmat)
					ntimes = phnmat.shape[1]
					print('phnmat shape is:')
					print(phnmat.shape)
				if binaryfeatmat:
					stim_dict[wav_name].append(phnfeatmat)
					print('phnfeatmat shape is:')
					print(phnfeatmat.shape)
				if envelope:
					envs = fh['/%s/%s/stim/envelope' %(stimulus_class, wav_name)][:] 
					envs = scipy.signal.resample(envs, ntimes) #resampling to size of phnfeat
					stim_dict[wav_name].append(envs.T)
					print('envs shape is:')
					print(envs.shape)
				if pitch:
					pitch_mat = fh['/%s/%s/stim/pitches' %(stimulus_class, wav_name)][:] 
					pitch_mat = scipy.signal.resample(pitch_mat, ntimes) #resample to size of phnfeat
					pitch_mat = np.atleast_2d(pitch_mat)
					stim_dict[wav_name].append(pitch_mat)
					print('pitch_mat shape is:')
					print(pitch_mat.shape)	
				if binned_pitches:
					binned_p = fh['/%s/%s/stim/binned_pitches' %(stimulus_class, wav_name)][:] 
					#binned_p = scipy.signal.resample(binned_p, ntimes) #resample to size of phnfeat
					binned_p = np.atleast_2d(binned_p)
					stim_dict[wav_name].append(binned_p.T)
					print('binned pitch shape is:')
					print(binned_p.shape)				
				if gabor_pc10:
					gabor_pc10_mat = fh['/%s/%s/stim/gabor_pc10' %(stimulus_class, wav_name)][:]
					stim_dict[wav_name].append(gabor_pc10_mat.T)
					print('gabor_mat shape is:')
					print(gabor_pc10_mat.shape)  
				if spectrogram:
					specs = fh['/%s/%s/stim/spec' %(stimulus_class, wav_name)][:] 
					specs = scipy.signal.resample(specs, ntimes, axis=1)
					new_freq = 15 #create new feature size, from 80 to 15. Easier to fit STRF with the specified time delay
					specs = scipy.signal.resample(specs, new_freq, axis=0)
					stim_dict[wav_name].append(specs)
					print('specs shape is:')
					print(specs.shape)
					freqs = fh['/%s/%s/stim/freqs' %(stimulus_class, wav_name)][:]
				if spectrogram_scaled:
					specs = fh['/%s/%s/stim/spec' %(stimulus_class, wav_name)][:] 
					specs = scipy.signal.resample(specs, ntimes, axis=1)
					new_freq = 15 #create new feature size, from 80 to 15. Easier to fit STRF with the specified time delay
					specs = scipy.signal.resample(specs, new_freq, axis=0)
					specs  = specs/np.abs(specs).max()
					stim_dict[wav_name].append(specs)
					print('specs shape is:')
					print(specs.shape)
				if scene_cut:
					s_cuts = fh['/%s/%s/stim/scene_cut' %(stimulus_class, wav_name)][:] 
					s_cuts = scipy.signal.resample(s_cuts, ntimes, axis=1)
					stim_dict[wav_name].append(s_cuts)
					print('scene cut shape is:')
					print(s_cuts.shape)
			
					#return freqs once
					freqs = fh['/%s/%s/stim/freqs' %(stimulus_class, wav_name)][:]
			except Exception:
				traceback.print_exc()
				
			if eeg_epochs:
				try: 
					epochs_data = fh['/%s/%s/resp/%s/epochs' %(stimulus_class, wav_name, subject)][:]
					if resp_mean:
						print('taking the mean across repeats')
						epochs_data = epochs_data.mean(0)
						epochs_data = scipy.signal.resample(epochs_data.T, ntimes).T #resample to size of phnfeat
					else:
						epochs_data = scipy.signal.resample(epochs_data, ntimes, axis=2)
					print(epochs_data.shape)
					resp_dict[wav_name].append(epochs_data)
					
				except Exception:
					traceback.print_exc()
					# print('%s does not have neural data for %s'%(subject, wav_name))

					# epochs_data = []

	if spectrogram:
		return resp_dict, stim_dict, freqs

	if spectrogram_scaled:
		return resp_dict, stim_dict, freqs
		
	else:
		return resp_dict, stim_dict


def strf_features(subject, stimulus_class, data_dir, full_gabor = False, full_audio_spec = True, full_model = True, pitchUenvs = False, pitchUphnfeat = False, 
	envsUphnfeat = False, phnfeat_only = False, envs_only = False, pitch_only = False, gabor_only = False, spec = False, pitchphnfeatspec = False, binned_pitch_full_audio = False, binned_pitch_envs=False, binned_pitch_phnfeat=False,
	binned_pitch_full_audiovisual=False, binned_pitch_only=False, pitchUspec=False, phnfeatUspec=False, spec_scaled=False, pitchphnfeatspec_scaled=False, scene_cut=False, scene_cut_gaborpc=False):
	"""
	Run your TRF or mTRF, depending on the number of features you input (phnfeat, envs, pitch)
	Test data is always set for TIMIT and Movie Trailers -- see stimulus_class
	
	Default STRF will always run the full model (phnfeat + pitch + envs) for each condition (TIMIT or MovieTrailers). 
	To change this, adjust the loadEEGh5 function to update True/False for the features (combination of features) you want to run. 


	Parameters
	----------
	subject : string 
		subject ID (i.e. MT0002)
	stimulus_class : string 
		MovieTrailers or TIMIT 
	full_model : bool
		- load envelope, phonological features, and pitch per subject for STRF analysis
		(default : True)
	pitchUenvs : bool
		- load phonological features and pitch per subject for STRF analysis (pair-wise models)
		(default : False)
	pitchUphnfeat : bool
		- load phonological features and pitch per subject for STRF analysis (pair-wise models)
		(default : False)
	envsUphnfeat : bool
		- load phonological features and envelope per subject for STRF analysis (pair-wise models)
		(default : False)
	phnfeat_only : bool 
		- only load phonological features from .h5 file
		(default : False)
	envs_only : bool 
		- only load acoustic envelope from .h5 file
		(default : False)
	pitch_only : bool 
		- only load pitch from .h5 file
		(default : False)
	gabor_only : bool 
		- only load gabor from .h5 file -- will only work for MovieTrailers
		(default : False)
	pitchphnfeatspec : bool 
		- load pitch (F0) + phonological features + spectrogram 
		(default : False)
	binned_pitch_full_audio : bool
		- binned pitch + phn feat + envelope -- full acoustic model 
		(default : False)
	binned_pitch_envs : bool 
		- pairwise features for binned pitch and acoustic envelope 
		(default : False)
	binned_pitch_phnfeat : bool
		- pairwise features for binned pitch and phonological features 
		(default : False)
	binned_pitches_audiovisual : bool  
		- full auditory and visual model 
		(default : False)
	binned_pitch_only : bool 
		- pitch feature only with binned pitch 
		(default : False)
	pitchUspec : bool 
		- pairwise F0 and spectrogram 
		(default : False)
	phnfeatUspec : bool 
		 - pairwise phonological features and spectrogram
		 (default : False)
	spec_scaled : bool 
		- spectrogram features which are scaled 
		(default : False)
	pitchphnfeatspec_scaled : bool 
		- a full auditory model which includes the F0 + phnfeats + scaled spectrogram features 
		(default : False)
	scene_cut : bool  
		- scene cut visual features only 
		(default : False)
	scene_cut_gaborpc : bool 
		- scene cut + gabor wavelet (10 features decomposed)
		(default : False)

	data_dir : string 
		-change this to match where .h5 is along with EEG data 


	Returns
	-------
	wt : numpy array
	corrs : numpy array
	valphas : numpy array
	allRcorrs : numpy array
	all_corrs_shuff : list

	"""	

	if full_audio_spec: #this is phn feat, envs, pitch, and spectrogram full audio model 
		resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, 
									 spectrogram=True, binned_pitch_full_audio=False, binned_pitch_full_audiovisual=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'pitchenvsphnfeatspec'

	if full_gabor: 
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=True, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'pitchenvsphnfeatgabor10pc'

	if full_model:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'pitchenvsphnfeat'

	if pitchUenvs: 
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=True, pitch=True, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'envspitch'

	if pitchUphnfeat: 
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'pitchphnfeat'

	if envsUphnfeat: 
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'envsphnfeat'

	if phnfeat_only: 
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'phnfeat'

	if envs_only: 
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'envs'

	if pitch_only: 
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'pitch'

	if gabor_only:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'gabor_only'
	
	if spec:
		resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=True, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'spec'
	
	if pitchphnfeatspec:
		resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, spectrogram=True, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		
		strf_output = 'pitchspecphnfeat'

	if binned_pitch_full_audio:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'binned_pitches'

	if binned_pitch_phnfeat:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binarymat=False, binaryfeatmat = True, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'binned_pitch_phnfeat'

	if binned_pitch_envs:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'binned_pitch_envs'

	if binned_pitch_full_audiovisual:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=True, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'binned_pitches_audiovisual'

	if binned_pitch_only:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'binned_pitch_only'

	if scene_cut:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=True)
		strf_output = 'scene_cut'

	if scene_cut_gaborpc:
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True, 
			binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=True)
		strf_output = 'scene_cut_gabor'

	if pitchUspec:
		resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, spectrogram=True, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'pitchspec'

	if phnfeatUspec:
		resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=True, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		strf_output = 'phnfeatspec'

	if pitchphnfeatspec_scaled:
		resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = True, binarymat=False, envelope=False, pitch=True, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=True, scene_cut=False)
		strf_output = 'phnfeatspec_scaled'

	if spec_scaled:
		resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
										 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=True, scene_cut=False)
		strf_output = 'spec_scaled'

	stim_list = []
	for key in resp_dict.keys():
		stim_list.append(key)
	# Populate stim and resp lists (these will be used to get tStim and tResp, or vStim and vResp) -- based on TIMIT or MT from loading h5 above
	stim = stim_dict 
	resp = resp_dict 

	# if MT0001:
	# 	test_set = ['paddington-2-trailer-1_a720p.wav'] #Just for MT0001 - set function input to True 
	
	if stimulus_class == 'TIMIT':
		test_set = ['fcaj0_si1479.wav', 'fcaj0_si1804.wav', 'fdfb0_si1948.wav', 
			'fdxw0_si2141.wav', 'fisb0_si2209.wav', 'mbbr0_si2315.wav', 
			'mdlc2_si2244.wav', 'mdls0_si998.wav', 'mjdh0_si1984.wav', 
			'mjmm0_si625.wav']

	elif stimulus_class == 'MovieTrailers':
		if subject == 'MT0001': #this is because MT0001 only heard one of the test set trailers, as opposed to both 
			test_set = ['paddington-2-trailer-1_a720p.wav']
		else:
			test_set = ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav']

	# else:
	# 	#test_set = ['paddington-2-trailer-1_a720p.wav'] #Just for MT0001
	# 	test_set = ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav'] #the test set for the remaining MTs
			
		
	all_stimuli = [k for k in (stim_dict).keys() if len(resp_dict[k]) > 0]
	training_set = np.setdiff1d(all_stimuli, test_set)
	training_set = training_set[training_set != 'the-lego-ninjago-movie-trailer-2_a720p.wav']
	print(training_set)
	
	val_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	train_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	for i in np.arange(len(all_stimuli)):
		if all_stimuli[i] in test_set:
			print(all_stimuli[i])
			val_inds[i] = True
		else:
			train_inds[i] = True

	print("Total number of training sentences:")
	print(sum(train_inds))
	print("Total number of validation sentences:")
	print(sum(val_inds))

	train_inds = np.where(train_inds==True)[0]
	val_inds = np.where(val_inds==True)[0]

	print("Training indices:")
	print(train_inds)
	print("Validation indices:")
	print(val_inds)
	
	# For logging compute times, debug messages
	
	logging.basicConfig(level=logging.DEBUG) 

	#time delays used in STRF
	delay_min = 0.0
	delay_max = 0.6
	wt_pad = 0.0 # Amount of padding for delays, since edge artifacts can make weights look weird

	fs = 128.0
	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) #create array to pass time delays in

	print("Delays:", delays)

	# Regularization parameters (alphas - also sometimes called lambda)
	alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

	nalphas = len(alphas)
	use_corr = True # Use correlation between predicted and validation set as metric for goodness of fit
	single_alpha = True # Use the same alpha value for all electrodes (helps with comparing across sensors)
	nboots = 20 # How many bootstraps to do. (This is number of times you take a subset of the training data to find the best ridge parameter)

	all_wts = [] # STRF weights (this will be ndelays x channels)
	all_corrs = [] # correlation performance of length [nchans]
	all_corrs_shuff = [] # List of lists, how good is a random model

	# train_inds and val_inds are defined in the cell above, and is based on specifying stimuli heard more than once, 
	# which will be in the test set, and the remaining stimuli will be in the training set 
	current_stim_list_train = np.array([all_stimuli[r][0] for r in train_inds])
	current_stim_list_val = np.array([all_stimuli[r][0] for r in val_inds])

	# Create training and validation response matrices
	print(resp_dict[training_set[0]][0].shape)
	print(test_set)
	
	print(len(training_set))
	for r in training_set:
		print(r)

	
	tResp = np.hstack([resp_dict[r][0] for r in training_set]).T
	vResp = np.hstack([resp_dict[r][0] for r in test_set]).T


	# Create training and validation stimulus matrices

	tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
	vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in test_set]))

	tStim_temp = tStim_temp/tStim_temp.max(0)
	vStim_temp = vStim_temp/vStim_temp.max(0)
	print('**********************************')
	print(tStim_temp.max(0).shape)
	print(vStim_temp.max(0).shape)
	print('**********************************')

	tStim = make_delayed(tStim_temp, delays)
	vStim = make_delayed(vStim_temp, delays)

	chunklen = np.int(len(delays)*3) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')

	nchans = tResp.shape[1] # Number of electrodes/sensors
	
	#plot neural data responses and binary feature matrix to identify timing for all phonological features based on stimulus 
	plt.figure(figsize=(15,3))
	elec = 15
	nsec_start = 390
	nsec_end = 410
	plt.figure(figsize=(10,5))
	plt.subplot(2,1,1)
	plt.plot(tResp[np.int(fs*nsec_start):np.int(fs*nsec_end),42]/tResp[:np.int(fs*nsec_start),elec].max(), 'r') #this is the response itself - EEG data (training)
	plt.subplot(2,1,2)
	plt.imshow(tStim_temp[np.int(fs*nsec_start):np.int(fs*nsec_end),:].T, aspect='auto', vmin=-1, vmax=1,interpolation='nearest', cmap=cm.RdBu) #envelope of trianing sound stimuli
	plt.colorbar()
	
	print('*************************')
	print(vStim)
	print('*************************')
	print('printing vResp: ')
	print(vResp)
	print('*************************')
	
	# Fit the STRFs
	wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = bootstrap_ridge(tStim, tResp, vStim, vResp, 
																		  alphas, nboots, chunklen, nchunks, 
																		  use_corr=use_corr,  single_alpha = single_alpha, 
																		  use_svd=False, corrmin = 0.05,
																		  joined=[np.array(np.arange(nchans))])

	print('*************************')
	print('pred value is: ')
	print(pred)
	print('*************************')
	print(wt.shape)
	#when wt padding is 0:
	if wt_pad > 0:

		good_delays = np.ones((len(delays), 1), dtype=np.bool)
		good_delays[np.where(delays<0)[0]] = False
		good_delays[np.where(delays>=np.ceil(delay_max*fs))[0]] = False
		good_delays = good_delays.ravel()



		print("Reshaping weight matrix to get rid of padding on either side")
		wt2 = wt.reshape((len(delays), -1, wt.shape[1]))[len(np.where(delays<0)[0]):-(len(np.where(delays<0)[0])),:,:]
		wt2 = wt2.reshape((wt2.shape[0]*wt2.shape[1], wt2.shape[2]))
	else:
		wt2 = wt

	print(wt2.shape)
	all_wts.append(wt2)
	all_corrs.append(corrs)
	

	plt.figure()
	# Plot correlations of model by electrode
	plt.plot(all_corrs[0])
	all_wts[0].shape[0]/14

	plt.figure()
	
	
	#BOOTSTRAPPING BELOW 

	# Determine whether the correlations we see for model performance are significant
	# by shuffling the data and re-fitting the models to see what "random" performance
	# would look like.
	#
	# How many bootstraps to do for determining bootstrap significance
	# The minimum p-value you can get from this is 1/nboots_shuffle
	# So for nboots_shuffle = 100, you can get p_values from 0.01 to 1.0
	nboots_shuffle = 100 

	nresp, nelecs = tStim.shape
	allinds = range(nresp)

	print("Determining significance of the correlation values using a bootstrap with %d iterations"%(nboots_shuffle))
	for n in np.arange(nboots_shuffle):
		print("Bootstrap %d/%d"%(n+1, nboots_shuffle))
		indchunks = list(zip(*[iter(allinds)]*chunklen))
		random.shuffle(indchunks)
		shuff_inds = list(itools.chain(*indchunks[:nchunks]))
		tStim_shuff = tStim.copy()
		tResp_shuff = tResp.copy()
		tStim_shuff = tStim_shuff[shuff_inds,:]
		tResp_shuff = tResp_shuff[:len(shuff_inds),:]

		corrs_shuff = eigridge_corr(tStim_shuff, vStim, tResp_shuff, vResp, [valphas[0]], corrmin = 0.05)
		all_corrs_shuff.append(corrs_shuff)

	# all_corrs_shuff is a list of length nboots_shuffle
	# Each element is the correlation for a random model for each of the 64 electrodes for that iteration
	# We use this to figure out [nboots_shuffle] possible values of random correlations for each electrode,
	# then use this to determine if the correlations we're actually measuring with the non-shuffled data are 
	# significantly higher than this
	
	
	# Get the p values of each of the significant correlations
	all_pvals = [] 

	all_c_s=np.vstack((all_corrs_shuff)) # Get the shuffled random correlations for this model

	# Is the correlation of the model greater than the shuffled correlation for random data?
	h_val = np.array([all_corrs[0] > all_c_s[c] for c in np.arange(len(all_c_s))])
	print(h_val.shape)

	# Count the number of times out of nboots_shuffle that the correlation is greater than 
	# random, subtract from 1 to get the bootstrapped p_val (one per electrode)
	p_val = 1-h_val.sum(0)/nboots_shuffle

	all_pvals.append(p_val)
	
	#load in your ICA data for your particular subject - will be used to fit significant responses on topo maps 
	raw = load_ICA_data(subject, data_dir, file='post_ICA', preload=True)
	if 'STI 014' in raw.info['ch_names']:
		raw.drop_channels(['vEOG', 'hEOG', 'STI 014'])
	else:
		raw.drop_channels(['vEOG', 'hEOG'])

	chnames = raw.info['ch_names']
	chnames = np.array(chnames)

	# Find the maximum correlation across the shuffled and real data
	max_corr = np.max(np.vstack((all_corrs_shuff[0], all_corrs[0])))+0.01 #why is this structured the way it is? Adding 0.01?

	# Plot the correlations for each channel separately
	plt.figure(figsize=(15,3))
	plt.plot(all_corrs[0])

	# Plot an * if the correlation is significantly higher than chance at p<0.05
	for i,p in enumerate(all_pvals[0]):
		if p<0.05:
			plt.text(i, max_corr, '*')

	# Plot the shuffled correlation distribution
	shuffle_mean = np.vstack((all_corrs_shuff)).mean(0) #correlation output form model -- which electrodes are correlated w/ each other, take average of this
	shuffle_stderr = np.vstack((all_corrs_shuff)).std(0)/np.sqrt(nboots_shuffle) #normalization of which electrodes are correlated w/ each other

	plt.fill_between(np.arange(nchans), shuffle_mean-shuffle_stderr, #normalization here
					 shuffle_mean+shuffle_stderr, color=[0.5, 0.5, 0.5])
	plt.plot(shuffle_mean, color='k')
	plt.gca().set_xticks(np.arange(len(all_corrs[0])))
	plt.gca().set_xticklabels(chnames, rotation=90);
	plt.xlabel('Channel')
	plt.ylabel('Model performance')
	plt.legend(['Actual data','Null distribution'])
	plt.savefig('%s/%s/%s_ch_distribution_%s.pdf' %(data_dir, subject, strf_output, stimulus_class)) #save fig

	
	#plot the significant correlations for participant on topo map 
	significant_corrs = np.array(all_corrs[0])
	significant_corrs[np.array(all_pvals[0])>0.05] = 0

	plt.figure(figsize=(5,5))
	print(['eeg']*2)
	info = mne.create_info(ch_names=raw.info['ch_names'][:64], sfreq=raw.info['sfreq'], ch_types=64*['eeg'])
	raw2 = mne.io.RawArray(np.zeros((64,10)), info)
	montage = mne.channels.read_montage('%s/montage/AP-128.bvef' %(data_dir), unit='mm')
	raw2.set_montage(montage) #set path for MNE montage file
	mne.viz.plot_topomap(significant_corrs, raw2.info, vmin=0, vmax=max_corr)
	#plt.savefig('%s/%s/%s_topomap_%s.png' %(data_dir, subject, strf_output, stimulus_class)) #save fig

	#plt.savefig('Topomap_MT.png')
	print(np.array(all_wts).shape)

	#save STRF as .h5 file based on condition type:
	if stimulus_class == 'TIMIT': 
		strf_file = '%s/%s/%s_STRF_by_%s_%s.hf5'%(data_dir, subject, subject, strf_output, stimulus_class)
		print("Saving file to %s"%(strf_file))
		with h5py.File(strf_file, 'w') as f:
			f.create_dataset('/wts_timit', data = np.array(all_wts[0])) #weights for MT/timit
			f.create_dataset('/corrs_timit', data = np.array(all_corrs[0])) #correlations for MT/timit
			f.create_dataset('/train_inds_timit', data = train_inds) #training sets for MT/timit
			f.create_dataset('/val_inds_timit', data = val_inds) #validation sets for MT (test set)/timit
			f.create_dataset('/pvals_timit', data = all_pvals) #save all pvals 
			f.create_dataset('/delays_timit', data = delays) #save delays 
			f.create_dataset('/valphas_timit', data = valphas) #save alpha value used for bootstrapping
			f.create_dataset('/allRcorrs_timit', data = allRcorrs) 
			f.create_dataset('/all_corrs_shuff_timit', data = all_corrs_shuff) 
	else:
		#strf_file = '%s/%s/%s_STRF_by_pitchenvsphnfeat_MT.hf5'%(data_dir, subject, subject)
		strf_file = '%s/%s/%s_STRF_by_%s_MT.hf5'%(data_dir, subject, subject, strf_output)
		print("Saving file to %s"%(strf_file))
		with h5py.File(strf_file, 'w') as f:
			f.create_dataset('/wts_mt', data = np.array(all_wts[0])) #weights for MT/timit
			f.create_dataset('/corrs_mt', data = np.array(all_corrs[0])) #correlations for MT/timit
			f.create_dataset('/train_inds_mt', data = train_inds) #training sets for MT/timit
			f.create_dataset('/val_inds_mt', data = val_inds) #validation sets for MT (test set)/timit
			f.create_dataset('/pvals_mt', data = all_pvals) #save all pvals 
			f.create_dataset('/delays_mt', data = delays) #save delays
			f.create_dataset('/valphas_mt', data = valphas) #save alpha value used for bootstrapping
			f.create_dataset('/allRcorrs_mt', data = allRcorrs) 
			f.create_dataset('/all_corrs_shuff_mt', data = all_corrs_shuff) #

	#THIS PLOT SHOWS THE DISTRIBUTION OF THE PREDICTED VS ACTUAL CORRELATIONS FOR EACH STIMULUS SET RUN
	np.vstack((all_corrs_shuff)).ravel().shape
	plt.figure()
	plt.hist(np.hstack((all_corrs_shuff)).ravel(), bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.hist(all_corrs[0], bins=np.arange(-0.2,max_corr,0.005), alpha=0.5, density=True)
	plt.xlabel('Model fits (r-values)')
	plt.ylabel('Number')
	plt.title('Correlation histograms')
	plt.legend(['Null distribution', 'EEG data'])
	plt.savefig('%s/%s/%s_corrHistogram_%s.pdf' %(data_dir, subject, stimulus_class, strf_output)) #save fig
	#Number of data points for a given bin that occurred 

	return wt, corrs, valphas, allRcorrs, all_corrs_shuff

def predict_response(wt, vStim, vResp):
	''' Predict the response to [vStim] given STRF weights [wt],
	compare to the actual response [vResp], and return the correlation
	between predicted and actual response.

	Inputs:
		wt: [features x delays] x electrodes, your STRF weights
		vStim: time x [features x delays], your delayed stimulus matrix
		vResp: time x electrodes, your true response to vStim
	Outputs:
		corr: correlation between predicted and actual response
		pred: prediction for each electrode [time x electrodes]
	'''
	nchans = wt.shape[1]
	print('Calculating prediction...')
	pred = np.dot(vStim, wt)

	print('Calculating correlation')
	corr = np.array([np.corrcoef(vResp[:,i], pred[:,i])[0,1] for i in np.arange(nchans)])

	return corr, pred

def create_subject_list(data_dir):

	'''
	Generates list of participants for analysis

	Parameters
	-----------
	data_dir : string 
		path to subject folders 

	Returns
	-------
		- list of subjects
	'''
	subject_list = os.listdir(data_dir)
	subject_list = [sub for sub in subject_list if 'MT' in sub and int(re.search(r'\d+', sub).group()) <= 17 and len(sub)<7]
	subject_list.sort()
	
	return subject_list

def convexHull(stimulus_class, subject_list, data_dir, save_dir, save_fig=False):

	"""
	Create a Convex-Hull plot of each stimulus condition to compare the full model to individual features. 
	Takes the average correlation values across participants and groups based on feature or full model to 
	depict model performance from mTRF 

	Uses the individual .h5 files created from the STRF output for each condition and loads the correlation values per subject
	Uses normalized correlation values form individual .h5 files 

	Parameters
	----------
	stimulus_class : string 
		MT or TIMIT 
	subject_list : list
		inputted from create_subject_list function 
	data_dir : string 
		-change this to match where all inidivdual .h5 files are  
	save_fig : bool 
		- outputs figure/plot when True
		(default : True)
	save_dir : string 
		-change this to match where to save the plot for each condition in PDF format 

	Returns
	-------
		- Convex Hull plot, saved in save_dir directory (PDF).

	"""
	
	fig = plt.figure(figsize=(15,9))
	ax = fig.subplots()
	if stimulus_class == 'TIMIT':
		print('TIMIT')
		plt.plot([-0.2, 1.0], [-0.2, 1.0], 'black', label='unity')

		
	elif stimulus_class == 'MT':
		print('trailers')
		plt.plot([-0.1, 0.5], [-0.1, 0.5], 'black', label='unity')

	else:
		print('Undefined stimulus class')

	corrs = []
	corrs_sig = []

	for idx, s in enumerate(subject_list):
		with h5py.File('%s/%s/%s_STRF_by_binned_pitches_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full auditory model
			fullModel = fh['corrs_%s_norm' %(stimulus_class.lower())][:] 
			cc_max = fh['cc_max_%s' %(stimulus_class.lower())][:] 
			fullModel[np.isinf(fullModel)]=0
			p_vals_full =  fh['pvals_%s' %(stimulus_class.lower())][:]
			full_sig = (fullModel[np.where(p_vals_full[0] < 0.05)])
		
			#binned pitch full model 
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #binned pitches only
			pitch = fh['corrs_%s_norm' %(stimulus_class.lower())][:]  
			pitch[np.isinf(pitch)]=0
			p_val_pitch =  fh['pvals_%s' %(stimulus_class.lower())][:]
			binned_full_sig = (pitch[np.where(p_vals_full[0] < 0.05)])


		with h5py.File('%s/%s/%s_STRF_by_envs_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #envs model only 
			envs = h['corrs_%s_norm' %(stimulus_class.lower())][:]
			envs[np.isinf(envs)]=0
			p_vals_envs = h['pvals_%s' %(stimulus_class.lower())][:]		
			envs_sig = (envs[np.where(p_vals_full[0] < 0.05)])

		with h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #phnfeat model only 
			phnfeat = h['corrs_%s_norm' %(stimulus_class.lower())][:]
			phnfeat[np.isinf(phnfeat)]=0
			p_vals_phnfeat = h['pvals_%s' %(stimulus_class.lower())][:]
			phnfeat_sig = (phnfeat[np.where(p_vals_full[0] < 0.05)])

		corrs.append([fullModel, envs, phnfeat, pitch])
		corrs_sig.append([full_sig, envs_sig,  phnfeat_sig, binned_full_sig])

	points = np.hstack(corrs).T
	points_sig = np.hstack(corrs_sig).T

	hull1=ConvexHull(np.vstack((points_sig[:,0] , points_sig[:,1])).T) #full model, envs
	hullv1 = hull1.vertices.copy()
	hullv1 = np.append(hullv1, hullv1[0])
	print(hullv1)

	hull2=ConvexHull(np.vstack((points_sig[:,0] , points_sig[:,2])).T) #full model, phnfeat
	hullv2 = hull2.vertices.copy()
	hullv2 = np.append(hullv2, hullv2[0])
	print(hullv2)

	hull3=ConvexHull(np.vstack((points_sig[:,0] , points_sig[:,3])).T) #full model, binnned pitch
	hullv3 = hull3.vertices.copy()
	hullv3 = np.append(hullv3, hullv3[0])
	print(hullv3)


	#fill between: 
	plt.fill(points_sig[hullv2,0], points_sig[hullv2,2], facecolor='#cd1a1e', alpha=0.4, zorder=3, label='full vs. phnfeat') #phnfeat
	plt.fill(points_sig[hullv3,0], points_sig[hullv3,3], facecolor='#64a7bc', alpha=0.4, zorder=1, label='full vs. pitch') #pitch 
	plt.fill(points_sig[hullv1,0], points_sig[hullv1,1], facecolor='#808080', alpha=0.4, zorder=2, label='full vs. envs') #envs

	plt.plot(points_sig[:,0], points_sig[:,1], '.', color='#808080', alpha=0.8) #envs
	plt.plot(points_sig[:,0], points_sig[:,2], '.', color='#cd1a1e', alpha=0.8) #phnfeat
	plt.plot(points_sig[:,0], points_sig[:,3], '.', color='#64a7bc',alpha=0.8) # binned pitch

	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
	plt.xlabel('Full model correlation values (r)')
	plt.ylabel('Individual model correlation values (r)')
	plt.title('Convex-Hull for %s (Feature distribution across subjects)' %(stimulus_class))
	plt.axis('tight')
	plt.axis('square')
	
	#save figure
	if save_fig:
		plt.savefig('%s/%s_ConvexHull.pdf' %(save_dir, stimulus_class))


def correct_rsqs(b, neg_only=True):
	"""
	Function to correct variance partitions for each feature 
	Accounts for negative values calculated from set theory and corrects values 

	Use corrected correlation values to create Venn Diagram and show unique variance 
	"""
		##  B . x = partition areas, where x is vector of model R^2 for models (A, B, C, AB, AC, BC, ABC)
	B = np.array([[0, 0, 0, 0, 0, -1, 1], # Abc: envelope 
				  [0, 0, 0, 0, -1, 0, 1], # aBc: phonological feature 
				  [0, 0, 0, -1, 0, 0, 1], # abC: pitch
				  [0, 0, -1, 0, 1, 1, -1], # ABc: envelope U phonological feature 
				  [0, -1, 0, 1, 0, 1, -1], # AbC: envelope U pitch
				  [-1, 0, 0, 1, 1, 0, -1], # aBC: phonological feature U pitch 
				  [1, 1, 1, -1, -1, -1, 1], # ABC: envelope U phonological feature U pitch 
				 ])
	#maxs = A.dot(np.nan_to_num(b))
	maxs = B.dot(np.nan_to_num(b))
	minfun = lambda x: (x ** 2).sum()
	#minfun = lambda x: np.abs(x).sum()

	biases = np.zeros((maxs.shape[1], 7)) + np.nan
	M = b.shape[1]
	for vi in range(M):
		if not (vi % 1000):
			print ("%d / %d" % (vi, M))
		
		if neg_only:
			bnds = [(None, 0)] * 7
		else:
			bnds = [(None, None)] * 7
		res = scipy.optimize.fmin_slsqp(minfun, np.zeros(7),
										#f_ieqcons=lambda x: maxs[:,vi] - A.dot(x),
										f_ieqcons=lambda x: maxs[:,vi] - B.dot(x),
										bounds=bnds, iprint=0)
		biases[vi] = res
	
	# compute fixed (legal) variance explained values for each model
	fixed_b = np.array(b) - np.array(biases).T

	orig_parts = B.dot(b)
	fixed_parts = B.dot(fixed_b)
	
	return biases, orig_parts, fixed_parts


def var_partition(stimulus_class, subject_list, data_dir, opacity=0.4):
	"""
	Calculate shared pair-wise or three-way variance across auditory feature types from calculated mTRFs.
	Output used to create a venn diagram 

	Parameters
	----------
	stimulus_class : string 
		- MT or TIMIT 
	subject_list : list
		- inputted from create_subject_list function 
	data_dir : string 
		-change this to match where all inidivdual .h5 files are 
	opacity : float
		- transparency for all colors in bar plot 
		(default: 0.4)
	"""
	
	variance_partition = []

	for idx, s in enumerate(subject_list):
		with h5py.File('%s/%s/%s_STRF_by_binned_pitches_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #full model
			corr_full = h['corrs_%s_norm' %(stimulus_class.lower())][:] 
			corr_full[np.isinf(corr_full)]=0
			p_val_full = h['pvals_%s' %(stimulus_class.lower())][:]  
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_envs_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate phnfeat model var
			corr_envspitch=h['corrs_%s_norm' %(stimulus_class.lower())][:]
			corr_envspitch[np.isinf(corr_envspitch)]=0
			#p_val_envspitch = h['pvals_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate envs model var:
			corr_pitchphnfeat=h['corrs_%s_norm' %(stimulus_class.lower())][:]
			corr_pitchphnfeat[np.isinf(corr_pitchphnfeat)]=0
			#p_val_pitchphnfeat = h['pvals_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_envsphnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate pitch model var
			corr_envsphnfeat=h['corrs_%s_norm' %(stimulus_class.lower())][:]
			corr_envsphnfeat[np.isinf(corr_envsphnfeat)]=0
			#p_val_envsphnfeat = h['pvals_%s'%(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_envs_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #envs model only 
			corr_envs = h['corrs_%s_norm' %(stimulus_class.lower())][:]
			corr_envs[np.isinf(corr_envs)]=0
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
			corr_pitch = h['corrs_%s_norm' %(stimulus_class.lower())][:]
			corr_pitch[np.isinf(corr_pitch)]=0
		with h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #phnfeat model only 
			corr_phnfeat = h['corrs_%s_norm' %(stimulus_class.lower())][:]
			corr_phnfeat[np.isinf(corr_phnfeat)]=0

		sig_corrs_full = ([np.where (p_val_full[0] < 0.05)])
		full_model = corr_full[sig_corrs_full] #sig elec corrs for full model 
		envs_U_pitch = corr_envspitch[sig_corrs_full]
		pitch_U_phnfeat = corr_pitchphnfeat[sig_corrs_full]
		envs_U_phnfeat = corr_envsphnfeat[sig_corrs_full]         
		envs_only = corr_envs[sig_corrs_full]
		pitch_only = corr_pitch[sig_corrs_full]
		phnfeat_only = corr_phnfeat[sig_corrs_full]
		
		variance_partition.append(np.array([envs_only, phnfeat_only, pitch_only, 
											envs_U_phnfeat, envs_U_pitch, pitch_U_phnfeat, full_model]))

	return variance_partition


def create_vennDiagram(subject_list, stimulus_class, variance_partition, save_dir, save_fig=True):
	"""
	Using the variation partition calculation between individual features, pair-wise features, and full auditory model, 
	create a venn diagram figure 

	Parameters
	----------
	subject_list : list
		- inputted from create_subject_list function 
	stimulus_class : string 
		MT or TIMIT 
	variance_partition : return function from var_partition function 
	save_fig : bool 
		- outputs figure/plot when True
		(default : True)
	save_dir : string 
		-change this to match where to save the plot for each condition in PDF format 

	Return
	------
	venn_diagram : list
	corr_fixed_parts : numpy array 
	"""

	rsq_corr = lambda c: (c ** 2) * np.sign(c) # r -> r^2 

	venn_diagram = []
	corr_fixed = []
	for idx, s in enumerate(subject_list):
		corr_est = np.squeeze(variance_partition[idx])

		# estimate biases, original partitions, and fixed partitions using r^2
		corr_biases, corr_orig_parts, corr_fixed_parts = correct_rsqs(rsq_corr(np.array(corr_est)), neg_only=False)
		print(corr_fixed_parts.shape)
		venn_diagram.append(corr_fixed_parts.mean(1))
		corr_fixed.append(corr_fixed_parts)

	venn_avg = np.array(venn_diagram).mean(0)

	v=venn3(subsets = (venn_avg[1], venn_avg[0], venn_avg[3], venn_avg[2], 
					  venn_avg[5], venn_avg[4], venn_avg[6]),
			set_labels = ('Phonological Features', 'Envelope', 'Pitch'), alpha=0.4)

	v.get_patch_by_id('100').set_color('#cd1a1e') #phnfeat only
	v.get_patch_by_id('001').set_color('#64a7bc') #pitch only
	v.get_patch_by_id('010').set_color('#808080') #envs only
	v.get_patch_by_id('101').set_color('#946774') #phnfeat + pitch intersection
	v.get_patch_by_id('110').set_color('#AA484B') #phnfeat + envs intersection
	v.get_patch_by_id('011').set_color('#73929B') #pitch + envs intersection
	v.get_patch_by_id('111').set_color('#996666') #all comb

	# #remove numbers from the venn diagram for each section 
	for idx, subset in enumerate(v.subset_labels):
		v.subset_labels[idx].set_visible(False)
	
	if save_fig:
			plt.savefig('%s/%s_vennDiagram.pdf' %(save_dir, stimulus_class))

	return venn_diagram, corr_fixed_parts

def feature_variance_Barplot(stimulus_class, subject_list, venn_diagram, save_dir, opacity=0.4,save_fig=False):
	"""
	Stacked bar plot demonstrating the amount of variance each feature across all subjects contributes to model performance.

	In other words, by adding a given speech feature representation into the combination of all three speech features 
	(acoustic + linguistic), what is the improvement in correlation value for each features and each subject? 

	Parameters
	----------
	subject_list : list
		- inputted from create_subject_list function 
	stimulus_class : string 
		MT or TIMIT 
	venn_diagram : return function from venn_diagram function 
	opacity : float
		- (default: 0.4)
	save_fig : bool 
		- outputs figure/plot when True
		(default : True)
	save_dir : string 
		-change this to match where to save the plot for each condition in PDF format 
	"""

	n_groups = len(subject_list)


	plt.figure(figsize=(8,4))


	index = np.arange(n_groups)
	bar_width = 0.25


	unique_var_pitch = []
	unique_var_phnfeat = []
	unique_var_envs = []

	for idx, s in enumerate(subject_list):
		unique_var_pitch.append(venn_diagram[idx][2]) #envs_U_phnfeat
		unique_var_phnfeat.append(venn_diagram[idx][1]) #envs_U_pitch
		unique_var_envs.append(venn_diagram[idx][0]) #pitch_U_phnfeat


	df = pd.DataFrame(list(zip(unique_var_pitch, unique_var_phnfeat, unique_var_envs, np.array(unique_var_pitch) + np.array(unique_var_phnfeat) + np.array(unique_var_envs))), 
		columns=['pitch', 'phnfeat', 'envs', 'sum']) #create dataframe from list
	df.index = subject_list 

	my_colors = ['#64a7bc', '#cd1a1e', '#808080']
	df.sort_values(by='sum', ascending=False)[['pitch', 'phnfeat', 'envs']].plot(kind='bar', stacked=True, color=my_colors, alpha=opacity, width=0.5)


	plt.xlabel('Participant number')
	plt.ylabel('Unique variance')
	plt.title('Unique feature variance (%s)' %(stimulus_class))

	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
	
	if save_fig:
		plt.savefig('%s/%s_uniqueVar.pdf' %(save_dir, stimulus_class))


def cross_prediction_ConvexHull(stimulus_class, subject_list, test_cond, train_cond, test_feature, train_feature, 
	data_dir, save_dir, opacity=0.4, save_fig=True):
	
	"""
	Scatter plots demonstrating if neural responses from one stimuli generalizes to another stimuli. In other words, 
	can you predict neural responses to movie trailers using the weights from TIMIT, and vice versa? 

	Parameters
	-----------
	data_dir : string
	subject_list : string
		- inputted from create_subject_list function - loops through all participants
	stimulus_class : string 
		- specify for type of plot size + corr max/min and plot label
		- either: TIMIT or MovieTrailers
		- This condition will be the same as test_cond (below)
	test_cond : string 
		- (either as TIMIT or MT) where train_cond will be the opposite condition
		- Specify condition, where the .h5 file correlations are the values used to train your model
		- original correlation values from .h5 file (for specific condition) will be used for plotting
	train_cond : string 
		- (either as TIMIT or MT) where test_cond will be the opposite condition
		- new correlation values will be predicted here and used for plotting=
	feature : string 
		- individual feature used for reading .h5 file and for training/testing 
		- defaults: (envs, phnfeat, binned_pitch_only, or binned_pitches for the full model])
			- (name of feature to run from .h5 file - must label exactly as .h5 name)
	opacity : float
		- transparency for all colors in bar plot 
		(default: 0.4)
	save_fig : bool 
		- outputs figure/plot when True
		(default : True)
	save_dir : string 
		-change this to match where to save the plot for each condition in PDF format 

	Returns
	-------
	Scatter plots showing cross prediction of conditions on an individual feature basis 
	p-value
	r-value 

	"""
	fig = plt.figure(figsize=(15,9))
	ax = fig.subplots()

	corrs = []
	corrs_sig = []
	corrs_nonsig = []

	corrs_train_less = []
	corrs_test_less = []
	#loop through subject list
	for idx, s in enumerate(subject_list):
		f = h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r')
		cc_max=f['/cc_max_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_%s_%s.hf5'%(data_dir, s, s, test_feature, test_cond), 'r') as fh:
			wts_test = fh['wts_%s' %(test_cond.lower())][:]
			corrs_test = fh['corrs_%s_norm' %(test_cond.lower())][:]
			cc_max = fh['cc_max_%s' %(test_cond.lower())][:]
			training_test = fh['train_inds_%s' %(test_cond.lower())][:]
			validation_test = fh['val_inds_%s' %(test_cond.lower())][:]
			pval_test = fh['pvals_%s' %(test_cond.lower())][:]
			test_nonsig = np.where(pval_test[0] > 0.05)
			test_sig = np.where(pval_test[0] < 0.05)
		with h5py.File('%s/%s/%s_STRF_by_%s_%s.hf5'%(data_dir, s, s, train_feature, train_cond), 'r') as h:
			wts_train = h['wts_%s' %(train_cond.lower())][:]
			corrs_train = h['corrs_%s' %(train_cond.lower())][:]/cc_max
			training_train = h['train_inds_%s' %(train_cond.lower())][:]
			validation_train = h['val_inds_%s' %(train_cond.lower())][:]

		print('*************************')
		print('Now processing %s' %(s))
		print('*************************')

		delay_min = 0.0
		delay_max = 0.6
		wt_pad = 0.0 # Amount of padding for delays, since edge artifacts can make weights look weird

		fs = 128.0
		delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) #create array to pass time delays in   
		
		if train_feature == 'phnfeat':
			plot_color = '#cd1a1e'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
																 binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		if train_feature == 'envs':
			plot_color = '#808080'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
																 binaryfeatmat = False, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)

		if train_feature == 'binned_pitch_only':
			plot_color = '#64a7bc'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
																 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
		if train_feature == 'binned_pitches':
			plot_color = '#996666'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
													 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
				#if stimulus_class == 'TIMIT':
		if train_feature == 'gabor_only':
			plot_color = '#6a0dad'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
													 binaryfeatmat=False, binarymat=False, envelope=False, pitch=False, gabor_pc10=True, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		if train_feature == 'audiovisual_gabor_sc':
			plot_color = '#fcba03'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
													 binaryfeatmat=True, binarymat=False, envelope=True, pitch=False, gabor_pc10=True, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=True)

			print("For the cross-prediction from MT to TIMIT for gabors, we remove the visual weights because TIMIT has no visual info")
			print(wts_test.shape)
			print(wts_train.shape)
			nfeats_test = np.int(wts_test.shape[0]/len(delays))
			print(nfeats_test)
			wts_train = wts_train.reshape(len(delays),-1,wts_train.shape[1])[:,:nfeats_test,:]
			wts_train = wts_train.reshape(-1, wts_train.shape[2])
			print(wts_train.shape)
			
		if test_feature == 'phnfeat':
			plot_color = '#cd1a1e'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
																 binaryfeatmat = True, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		if test_feature == 'envs':
			plot_color = '#808080'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
																 binaryfeatmat = False, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False)
		if test_feature == 'binned_pitch_only':
			plot_color = '#64a7bc'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
																 binaryfeatmat = False, binarymat=False, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)
		if test_feature == 'binned_pitches':
			plot_color = '#996666'
			resp_dict, stim_dict = loadEEGh5(s, stimulus_class, data_dir, eeg_epochs=True, resp_mean = True,
													 binaryfeatmat = True, binarymat=False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)


		#check that all resp_dict/neural epochs exist for all participants 
		all_stimuli = []
		for i in resp_dict.keys():
			x = (i if resp_dict[i] else 'False')
			if x != 'False':
				all_stimuli.append(x)
		vResp = np.hstack([resp_dict[r][0] for r in [all_stimuli[m] for m in validation_test]]).T
		

		# Create validation stimulus matrices

		if test_feature == 'binned_pitches':
			vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in [all_stimuli[m] for m in validation_test]]))
		else:
			vStim_temp = np.atleast_2d(np.vstack([stim_dict[r][0].T for r in [all_stimuli[m] for m in validation_test]])) 
		vStim = make_delayed(vStim_temp, delays)
		print('******************************')
		print(wts_train.shape)
		print(vStim.shape)
		print(vResp.shape)
		print('******************************')
		test_condition_corrs, train_condition_pred = predict_response(wts_train, vStim, vResp) 
		test_condition_corrs = test_condition_corrs/cc_max #divide test condition normalized correlation by the max corr value calculated from noise ceiling analysis 

		#load channel names from ICAed EEG data
		ds = load_ICA_data(s, data_dir)
		if 'STI 014' in ds.info['ch_names']:
			ds.drop_channels(['vEOG', 'hEOG', 'STI 014'])
		else:
			ds.drop_channels(['vEOG', 'hEOG'])

		chnames = ds.info['ch_names']
		chnames = np.array(chnames)

		corrs_train_less.append(chnames[np.where(corrs_test < 0)[0]])
		corrs_test_less.append(chnames[np.where(test_condition_corrs < 0)[0]])

		corrs.append([corrs_test, test_condition_corrs])
		corrs_nonsig.append([corrs_test[test_nonsig], test_condition_corrs[test_nonsig]])
		corrs_sig.append([corrs_test[test_sig], test_condition_corrs[test_sig]])
	
	points = np.hstack(corrs).T

	hull1=ConvexHull(np.vstack((points[:,0] , points[:,1])).T)
	hullv1 = hull1.vertices.copy()
	hullv1 = np.append(hullv1, hullv1[0])
	print(hullv1)


	#plot ConvexHull
	for idx, m in enumerate(subject_list):
		plt.plot(corrs_sig[idx][0], corrs_sig[idx][1], '.', color=plot_color, alpha=0.8)
		plt.plot(corrs_nonsig[idx][0], corrs_nonsig[idx][1], '.', color='#DCDBDB', alpha=0.7)

	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

	plt.xlabel('Predict %s from %s' %(test_cond, test_cond))
	plt.ylabel('Predict %s from %s' %(test_cond, train_cond))
	plt.axis('square')
	plt.axvline()
	plt.axhline()

	#plot regression line:
	if stimulus_class == 'TIMIT':
		plt.plot([-0.75, 1.0], [-0.75, 1.0], 'black', label='unity')
		[slope, intercept, r_value, p_value, std_err] = scipy.stats.linregress(corrs[0][0], corrs[0][1])
		plt.plot([-0.75,1.0], [-0.75*slope + intercept, 1.0*slope+intercept], color='red')
		print(r_value) #correlation with person 
		print(p_value) #high sigificant 

	#set axes ticks
	if stimulus_class == 'MovieTrailers':
		plt.plot([-0.3, 0.4], [-0.3, 0.4], 'black', label='unity') #
		plt.gca().set_xticks([-0.3, -0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4])
		plt.gca().set_yticks([-0.3, -0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4])

		[slope, intercept, r_value, p_value, std_err] = scipy.stats.linregress(corrs[0][0], corrs[0][1])
		plt.plot([-0.3,1.0], [-0.3*slope + intercept, 1.0*slope+intercept], color='red')
		print(r_value) #correlation with person 
		print(p_value) #high sigificant 


	if save_fig:
		plt.savefig('%s/%s_crossPred_%s.pdf' %(save_dir, stimulus_class, train_feature))

	return corrs, corrs_sig, corrs_train_less, corrs_test_less

def ANOVA_stats(subject_list, data_dir, h5_type, model_types):
	"""
	Perform non-parametric Friedman ANOVA test on specified model types
	Perform post-hoc tests of Friedman ANOVA is significant 
	
	Parameter
	---------
	subject_list : string
		- inputted from create_subject_list function - loops through all participants
	data_dir : string
	h5_type : string 
		- specify either TIMIT or MT to read into correlation values based on condition in .h5 file
	model_types : list with string
		- array (list) specified with individual or full feature models
		
	Return
	------
	Pandas dataframe and friedman ANOVA value
	
	If Friedman ANOVA, p<0.05, run post-hoc tests
	Will also return post-hoc test results
	
	"""
	
	all_subjs = []
	all_models = []
	all_corrs = []
	corrs = dict()

	for model in model_types: # 3 total models we are comparing
		for s in subject_list:
			# Load the STRF file for each individual model for the subject of interest
			# (phnfeat only, env only, or pitch only)
			strf_file = '%s/%s/%s_STRF_by_%s_%s.hf5'%(data_dir, s, s, model, h5_type) # The STRF for this subject and this model type (env, phnfeat, or binned_pitch_only)
			with h5py.File(strf_file,'r') as hf:
				corrs[s] = hf['corrs_%s_norm' %(h5_type.lower())][:] # Load the corrs
			for ch in np.arange(64):
				# We have to do this so we have the subjects and models
				# columns that match the correlations vector
				all_subjs.append(s)
				all_models.append(model)
				all_corrs.append(corrs[s][ch])
	data= {'corrs': np.array(all_corrs).ravel(), 'subject': all_subjs, 'STRF_type': all_models}
	df = pd.DataFrame.from_dict(data)
	df
	
	# Run a Friedman ANOVA (non-parametric equivalent of the repeated measures ANOVA)
	# with STRF performance as yhour dependent variable, STRF type (env, phnfeat, pitch) 
	# as your within subjects measure, and subject as your subject. Look at p-unc for
	# the p value
	data = df.groupby(['subject', 'STRF_type']).mean().reset_index()
	#print(data)
	pg.friedman(data=df, dv='corrs', within='STRF_type', subject='subject')
	
	# if p<0.05, run post-hoc sign-rank tests

	#extract just the corr values from the dataframe - will be used for post-hoc sign-rank tests
	pitch_x = data['corrs'][np.where(data['STRF_type']=='binned_pitch_only')[0]]
	phnfeat_x = data['corrs'][np.where(data['STRF_type']=='phnfeat')[0]]
	envs_x = data['corrs'][np.where(data['STRF_type']=='envs')[0]]
	totalmodel_x = data['corrs'][np.where(data['STRF_type']=='binned_pitches')[0]]


	#run wilcoxon signrank test - compare total model with individual features
	print('Full auditory model vs. phonological features: ')
	print(pg.wilcoxon(totalmodel_x, phnfeat_x, tail='two-sided')) 
	print('Full auditory model vs. acoustic envelope: ')
	print(pg.wilcoxon(totalmodel_x, envs_x, tail='two-sided')) 
	print('Full auditory model vs. binned pitches: ')
	print(pg.wilcoxon(totalmodel_x, pitch_x, tail='two-sided'))

	#run wilcoxon signrank test - compare individual feature models with each other 
	print('phonological features vs. binned pitches: ')
	print(pg.wilcoxon(phnfeat_x,pitch_x, tail='two-sided'))
	print('acoustic envelope vs. binned pitches: ')
	print(pg.wilcoxon(envs_x, pitch_x, tail='two-sided'))
	print('phonological features vs. acoustic envelope: ')
	print(pg.wilcoxon(phnfeat_x, envs_x, tail='two-sided')) 


def num_of_stim_reps(subject, stimulus_class, data_dir, save_dir, save_fig=True):

	"""
	Uses training data from Movie Trailer and TIMIT conditions to plot the amount of testing data, corresponding 
	with the average correlation value per repetition of test set. 


	Parameters:
	-----------
	data_dir : string
	save_dir : string
	stimulus_class : string 
		- specify for type of plot size + corr max/min and plot label
		- either: TIMIT or MovieTrailers
		- This condition will be the same as test_cond (below)
	save_fig : bool
		- Change to save the output figure 
		(default : True)

	Output
	-------
	- plot 
	- corrs_reps : .h5 file
		- outputs .h5 file with correlation values for bootstrap as individual layer 
	"""
	if stimulus_class == 'TIMIT':
		resp_dict, stim_dict = loadEEGh5(subject, 'TIMIT', data_dir, eeg_epochs=True, resp_mean = False, binarymat=False, 
		binaryfeatmat = True, envelope=True, pitch=True, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False) #load full model. Do not average across trials. 
		test_set = ['fcaj0_si1479.wav', 'fcaj0_si1804.wav', 'fdfb0_si1948.wav', 
		'fdxw0_si2141.wav', 'fisb0_si2209.wav', 'mbbr0_si2315.wav', 
		'mdlc2_si2244.wav', 'mdls0_si998.wav', 'mjdh0_si1984.wav', 
		'mjmm0_si625.wav']
		
		stim_list = []
		for key in resp_dict.keys():
			print(key)
			stim_list.append(key)
		all_stimuli = [k for k in stim_list if len(resp_dict[k]) > 0]
		training_set = np.setdiff1d(all_stimuli, test_set)
		print(training_set)


	if stimulus_class == 'MovieTrailers':
		print('Subject must be MT0020')

		resp_dict, stim_dict = loadEEGh5(subject, 'MovieTrailers', data_dir, eeg_epochs=True, resp_mean = False, binarymat=False, 
		binaryfeatmat = True, envelope=True, pitch=True, gabor_pc10=False, spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=False) #load full model. Do not average across trials. 

		
		trailers_list = ['angrybirds-tlr1_a720p.wav', 'bighero6-tlr1_a720p.wav', 'bighero6-tlr2_a720p.wav', 
		'bighero6-tlr3_a720p.wav', 'cars-3-trailer-4_a720p.wav', 'coco-trailer-1_a720p.wav', 
		'ferdinand-trailer-2_a720p.wav', 'ferdinand-trailer-3_a720p.wav', 'ice-dragon-trailer-1_a720p.wav', 
		'incredibles-2-trailer-1_a720p.wav', 'incredibles-2-trailer-2_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav',
		'insideout-usca-tlr2_a720p.wav', 'moana-clip-youre-welcome_a720p.wav', 'paddington-2-trailer-1_a720p.wav', 
		'pandas-trailer-2_a720p.wav', 'pele-tlr1_a720p.wav', 'the-breadwinner-trailer-1_a720p.wav', 
		'the-lego-ninjago-movie-trailer-1_a720p.wav', 'the-lego-ninjago-movie-trailer-2_a720p.wav', 
		'thelittleprince-tlr_a720p.wav', 'trolls-tlr1_a720p.wav']

		# resp_dict = {}

		# for k in trailers_list:
		# 	resp_dict[k] = [np.concatenate((resp_dict_MT0002[k][0], resp_dict_MT0020[k][0]), axis=0)]
		test_set = ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav'] #the test set for the remaining MTs

		all_stimuli = trailers_list
		training_set = np.setdiff1d(all_stimuli, test_set)
		print(training_set)

	val_inds = np.zeros((len(all_stimuli),), dtype=np.bool) 
	train_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
	for i in np.arange(len(all_stimuli)):
		if all_stimuli[i] in test_set:
			print(all_stimuli[i])
			val_inds[i] = True
		else:
			train_inds[i] = True

	print("Total number of training sentences:")
	print(sum(train_inds))
	print("Total number of validation sentences:")
	print(sum(val_inds))

	train_inds = np.where(train_inds==True)[0]
	val_inds = np.where(val_inds==True)[0]

	print("Training indices:")
	print(train_inds)
	print("Validation indices:")
	print(val_inds)

	# For logging compute times, debug messages

	logging.basicConfig(level=logging.DEBUG) 

	#time delays used in STRF
	delay_min = 0.0
	delay_max = 0.6
	wt_pad = 0.1 # Amount of padding for delays, since edge artifacts can make weights look weird

	fs = 128.0
	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) #create array to pass time delays in

	print("Delays:", delays)

	# Regularization parameters (alphas - also sometimes called lambda)
	alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

	nalphas = len(alphas)
	use_corr = True # Use correlation between predicted and validation set as metric for goodness of fit
	single_alpha = True # Use the same alpha value for all electrodes (helps with comparing across sensors)
	nboots = 20 # How many bootstraps to do. (This is number of times you take a subset of the training data to find the best ridge parameter)

	all_wts = [] # STRF weights (this will be ndelays x channels)
	all_corrs = [] # correlation performance of length [nchans]
	all_corrs_shuff = [] # List of lists, how good is a random model

	# train_inds and val_inds are defined in the cell above, and is based on getting 80% of the trials
	# for each unique stimulus to be in the training set, and the remaining 20% to be in 
	# the validation set
	current_stim_list_train = np.array([all_stimuli[r][0] for r in train_inds])
	current_stim_list_val = np.array([all_stimuli[r][0] for r in val_inds])

	# Create training and validation response matrices
	print(resp_dict[training_set[0]][0].shape)
	print(test_set)

	print(len(training_set))
	for r in training_set:
		print(r)


	# tResp = np.hstack([resp_dict[r][0] for r in training_set]).T
	# vResp = np.hstack([resp_dict[r][0] for r in test_set]).T


	# Create training and validation stimulus matrices

	tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
	vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in test_set if resp_dict[r][0].shape[0] >= 10]))
	tStim_temp = tStim_temp/tStim_temp.max(0)
	vStim_temp = vStim_temp/vStim_temp.max(0)
	print('**********************************')
	print(tStim_temp.max(0).shape)
	print(vStim_temp.max(0).shape)
	print('**********************************')

	tStim = make_delayed(tStim_temp, delays)
	vStim = make_delayed(vStim_temp, delays)

	chunklen = np.int(len(delays)*3) # We will randomize the data in chunks 
	nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')


	
	vResp_numtrials = [resp_dict[r][0].shape[0] for r in test_set if resp_dict[r][0].shape[0] >= 10]
	#print(vResp)
	print(vResp_numtrials)
	ntrials = np.min(vResp_numtrials)
	print(ntrials)
	
	vreps = np.arange(1,ntrials+1) # From 1 to 10 repeats of the validation set
	nboots = 10
	corrs_reps = dict()
	for v in vreps:
		corrs_reps[v] = []
		print('*****************************')
		print('Now on repetition # %d' %(v))
		print('*****************************')
		tResp = np.hstack([resp_dict[r][0].mean(0) for r in training_set]).T
		nchans = tResp.shape[1] # Number of electrodes/sensors
		
		trial_combos = [k for k in itools.combinations(np.arange(ntrials), v)]
		for t in trial_combos:
			vResp_temp = [resp_dict[r][0][t,:,:].mean(0) for r in test_set if resp_dict[r][0].shape[0] >= 10]

			vResp = np.hstack((vResp_temp)).T
			print(vResp.shape)
			# Fit the STRFs - RUNNING THE MODEL HERE!
			wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = bootstrap_ridge(tStim, tResp, vStim, vResp, 
																			  alphas, nboots, chunklen, nchunks, 
																			  use_corr=use_corr,  single_alpha = single_alpha, 
																			  use_svd=False, corrmin = 0.05,
																			  joined=[np.array(np.arange(nchans))])
			corrs_reps[v].append(corrs)
			plt.plot(v, corrs.mean(), '.')
			
	plt.xlabel('Number repeats')
	plt.ylabel('Average corr')

	corrs_reps_avg = []
	corrs_reps_std = []
	for i in np.arange(1, ntrials+1):
		corrs_reps_avg.append(np.mean(corrs_reps[i]))
		print(np.array(corrs_reps[i])[:,23])
		corrs_reps_std.append(np.std(corrs_reps[i])/np.sqrt(len(vreps)))


	plt.fill_between(vreps, np.array(corrs_reps_avg)+np.array(corrs_reps_std), np.array(corrs_reps_avg)-np.array(corrs_reps_std), alpha=0.5)

	plt.plot(vreps, corrs_reps_avg)

	plt.xlabel('Number repeats')
	plt.ylabel('Average validation correlation')

	#save corrs_reps into .h5 file for each repetition as individual key. Will save out as either TIMIT or MT reps
	with h5py.File('%s/%s_corrs_reps.hf5' %(data_dir, stimulus_class), 'w') as g:
		for idx, s in enumerate(corrs_reps.items()): 
			g.create_dataset('/rep%d' %(idx), data=np.array(s[1]))

	if save_fig:
		plt.savefig('%s/%s_10bootStReps.pdf' %(save_dir, stimulus_class))

	return corrs_reps
#make this channel by channel (as subplots for MT0002/MT0020)

def plot_test_set_reps(data_dir, save_dir, stimulus_class, subject='MT0002', opacity=0.4):
	if subject == 'MT0002' :
		nreps = 10 
		stimulus_class = 'TIMIT'

	elif subject == 'MT0020':
		nreps = 10 #or nreps=2
		stimulus_class = 'MT'
	f = h5py.File('%s/%s/%s_STRF_by_binned_pitches_%s.hf5' %(data_dir, subject, subject, stimulus_class), 'r')
	cc_max = f['/cc_max_%s' %(stimulus_class.lower())][:]


	all_corrs = np.zeros((nreps, 64))
	corrs_avg = []
	corrs_std = []
	all_corrs = []


	with h5py.File('%s/%s_%s_corrs_reps.hf5' %(data_dir, subject, stimulus_class), 'r') as fh:
		reps = np.arange(len(fh))
		reps_list=reps.tolist()
		for i in reps:
			corrs=[k for k in fh['/rep%d' %(i)]]
			#print(corrs)
			corrs=corrs/cc_max
			all_corrs.append(corrs)
			#print(corrs)
			
			####
			#corrs_avg.append(np.mean(corrs)) #append avg from corrs 
			#corrs_std.append(np.std(corrs)/np.sqrt(len(corrs))) #standard error 

			#####
			corrs_avg.append(corrs.mean(0).mean(0))
			corrs_std.append(corrs.mean(0).std(0)/np.sqrt(corrs.shape[1]))

		#all_corrs[:,idx]=np.array(corrs_avg)
		corrs_array = np.array(corrs_avg)
		#all_corrs[:corrs_array.shape[0]] = corrs_array

	plt.fill_between(np.arange(1,nreps+1), np.array(corrs_avg)+np.array(corrs_std), np.array(corrs_avg)-np.array(corrs_std), alpha=0.5)
	plt.plot(np.arange(1,nreps+1), corrs_avg)
	plt.title('%s %s amount of testing data' %(subject, stimulus_class))
	plt.xlabel('Number of repetitions')
	plt.ylabel('Average correlation (r')
	plt.savefig('%s/%s_%s_testsetreps.pdf' %(save_dir, subject, stimulus_class))

	if subject == 'MT0020':
			
		# performance improvement - Bar plot
		reps_list = ['2 reps', '10 reps']
		avg_corrs_MT0020 = []
		corrs_std_MT0020 = []

		avg_corrs_MT0020.append(corrs_avg[1])
		avg_corrs_MT0020.append(corrs_avg[9])
		corrs_std_MT0020.append(corrs_std[1])
		corrs_std_MT0020.append(corrs_std[9])

		n_groups = 2
		plt.figure(figsize=(8,4))
		bar_width = 0.5
		index = np.arange(n_groups)

		plt.bar(index, avg_corrs_MT0020, bar_width, yerr=corrs_std_MT0020, color='#996666', alpha=0.6)
		plt.xticks(np.arange(len(reps_list)), reps_list)
		plt.ylabel('Correlation Values')
		plt.title('Avg. correlation values MT0020 (two vs 10 reps)')
		plt.savefig('%s/%s_%s_reps_comparison.pdf' %(save_dir, subject, stimulus_class))

		#make convex-Hull for MT0020 based on average of nreps=2 and nreps=10

		plt.figure(figsize=(6,4))
		MT0020_corrs = []
		MT0020_corrs.append([all_corrs[9].mean(0), all_corrs[1].mean(0)]) #take last rep and 2nd rep and average across 0th dimension
		points = np.hstack(MT0020_corrs).T
		hull1=ConvexHull(np.vstack((points[:,0] , points[:,1])).T)
		hullv1 = hull1.vertices.copy()
		hullv1 = np.append(hullv1, hullv1[0])
		print(hullv1)
		plt.fill(points[hullv1,0], points[hullv1,1], facecolor='#9a6767', alpha=opacity, zorder=2) 
		plt.plot(all_corrs[9].mean(0),  all_corrs[1].mean(0), '.', color='#9a6767')
		plt.plot([0.0, 0.2], [0.0, 0.20], 'black', label='unity')
		plt.xlabel('MT0020 10 reps')
		plt.ylabel('MT0020 2 reps')
		plt.savefig('%s/%s_repNum_Hull_comparison.pdf' %(save_dir, subject))



def trailer_aud_segmentation(subject_list, data_dir, ts, stim_type, sfreq=128.0, delay_max=0.6, delay_min=0.0, timepoints=None):
	"""
	subj : string
	ts : string
		- movie trailer stimulus name to input from test set (either 'insideout-tlr2zzyy32_a720p.wav' or 'paddington-2-trailer-1_a720p.wav')
		(default : 'insideout-tlr2zzyy32_a720p.wav')
	stim_type : string
		- type of feature to correlation predicted vs. actual
		- Valid inputs: phnfeat, envs, binned_pitch_only, binned_pitches 
	sfreq : float 
		- sampling rate from EEG in Hz 
		(default : 128.0)
	delay_min : float 
		- time delay for STRFs, initial time (ms)
		(default : 0.0)
	delay_max : float
		- amount of time after onset of stimulus from EEG for STRFs (ms)
		(default : 0.6)
	nsec : int
		- how long is the sliding scale window to compare correlations (seconds)
		(default : 3)
	timepoints: np.array
		- speech, speech_w_noise, or noise only time sample indicies 
	"""
	mt_file = '%s/fullEEGmatrix.hf5' %(data_dir)

	mf=h5py.File(mt_file, 'r')
	phnfeat = mf['MovieTrailers/%s/stim/phn_feat_timings'%ts][:].T
	envelope = mf['MovieTrailers/%s/stim/envelope'%ts][:]
	binned_pitch = mf['MovieTrailers/%s/stim/binned_pitches'%ts][:]
	
	ntimes=phnfeat.shape[0]
	print('The shape for phonological features/ntimes is: %s' %(ntimes))

	#resample to shape phnfeat
	envs = scipy.signal.resample(envelope, ntimes)
	binned_pitch=scipy.signal.resample(binned_pitch,ntimes)
	binned_pitch=np.atleast_2d(binned_pitch)

	resp = []
	pred = []
	for idx, subject in enumerate(subject_list):
		resp.append(mf['MovieTrailers/%s/resp/%s/epochs'%(ts, subject)][:].mean(0).T)
		if stim_type == 'envelope':
			stims = envs # np.hstack((envelope, phnfeat))
			plot_color = '#808r080'
			f = h5py.File('%s/%s/%s_STRF_by_envs_MT.hf5' %(data_dir, subject, subject),'r') 
		elif stim_type == 'phnfeat':
			stims = phnfeat
			plot_color = '#cd1a1e'
			f = h5py.File('%s/%s/%s_STRF_by_phnfeat_MT.hf5' %(data_dir, subject, subject),'r') 
		elif stim_type == 'binned_pitch_only':
			stims = binned_pitch
			plot_color = '#64a7bc'
			f = h5py.File('%s/%s/%s_STRF_by_binned_pitch_only_MT.hf5' %(data_dir, subject, subject),'r') 
		elif stim_type == 'binned_pitches':
			stims = np.hstack((phnfeat, envs, binned_pitch)) #make sure this is in the right order!
			print('running new full model in correct order!')
			plot_color = '#996666'
			f = h5py.File('%s/%s/%s_STRF_by_binned_pitches_MT.hf5' %(data_dir, subject, subject),'r') 
		print('The shape for specific %s is %s: ' %(stim_type, stims.shape))
		#print(stims.shape)

		#f = h5py.File('%s/%s/%s_STRF_by_pitchenvsphnfeat_MT.hf5'%(data_dir, subj, subj), 'r')

		delays = np.arange(np.floor((delay_min)*sfreq), np.ceil((delay_max)*sfreq), dtype=np.int) #create array to pass time delays in
		

		if timepoints is not None:
			vStim = make_delayed(stims[timepoints,:], delays)
		else:
			vStim = make_delayed(stims, delays)
		#vStim_old = make_delayed(stims_old, delays)

		#load wt from subject-specific .h5 file
		wt = f['wts_mt'][:]
		#print(wt.shape)
		p_val = f['pvals_mt'][:]
		corrs = f['corrs_mt_norm'][:]
		cc_max = f['cc_max_mt'][:]
		corrs = corrs/cc_max


		pred.append(np.dot(vStim, wt))

	resp = np.array(resp)

	if timepoints is not None:
		resp = resp[:,timepoints,:]
	pred = np.array(pred)

	return resp, pred 

grid_dir = '/Users/maansidesai/Box/MovieTrailersTask/Data/TextGrids/'
#grid_name = 'insideout-tlr2zzyy32_a720p_corrected.TextGrid'

def auditory_environment_trailers_correlation(grid_dir, subject_list, data_dir, save_dir, grid_name = 'insideout-tlr2zzyy32_a720p_corrected.TextGrid', 
	ts='insideout-tlr2zzyy32_a720p.wav', ntimes=17364):

	'''
	For paddington:
	grid_name = 'paddington-2-trailer-1_a720p_corrected_spmubg.TextGrid'
	ntimes=17427
	ts='paddington-2-trailer-1_a720p.wav'

	For insideout:
	grid_name = 'insideout-tlr2zzyy32_a720p_corrected.TextGrid'
	ntimes = 17364
	ts = 'insideout-tlr2zzyy32_a720p.wav'
	'''
	clean_vs_noisy_speech = speech_nonspeech_tier_alignment_slidingCorr(grid_dir, grid_name, ntimes, sfreq=128.0)


	speech = np.where(clean_vs_noisy_speech == 1)[0]
	speech_w_noise = np.where(clean_vs_noisy_speech == 2)[0] #speech w/ noise
	noise = np.where(clean_vs_noisy_speech == 3)[0]

	stim_type = ['phnfeat', 'envelope', 'binned_pitch_only', 'binned_pitches']
	environment = ['speech', 'speech_w_noise', 'noise']

	corrs_speech =np.zeros((len(subject_list), 64,len(stim_type))) #[64 channels for phn feat] [64 ch for envs] 
	corrs_speech_noise = np.zeros((len(subject_list), 64,len(stim_type)))
	corrs_noise = np.zeros((len(subject_list), 64,len(stim_type)))

	for j, z in enumerate(stim_type):
		for subject, t in enumerate(subject_list): 
		#loop through each subject's resp info
		# for subject in np.arange(resp.shape[0]):
			speech_only_resp, speech_only_pred = trailer_aud_segmentation([t], data_dir, ts=ts, stim_type=z, sfreq=128.0, 
			delay_max=0.6, delay_min=0.0, timepoints=speech) 

			speech_noise_resp, speech_noise_pred = trailer_aud_segmentation([t], data_dir, ts=ts, stim_type=z, sfreq=128.0, 
			delay_max=0.6, delay_min=0.0, timepoints=speech_w_noise) 

			noise_resp, noise_pred = trailer_aud_segmentation([t], data_dir, ts=ts, stim_type=z, sfreq=128.0, 
			delay_max=0.6, delay_min=0.0, timepoints=noise) 

			#loop through 64 channels:
			for i in np.arange(64):
				corrs_speech[subject, i,j] = np.corrcoef(speech_only_pred[0,:,i], speech_only_resp[0,:,i])[0,1]
				corrs_speech_noise[subject, i,j] = np.corrcoef(speech_noise_pred[0,:,i], speech_noise_resp[0,:,i])[0,1]
				corrs_noise[subject, i,j] = np.corrcoef(noise_pred[0,:,i], noise_resp[0,:,i])[0,1]

	#plot avg correlation and standard error
	plt.figure(figsize=(8,4))
	features = ['speech', 'speech w/ noise', 'noise']
	#colors = ['#2f87af', '#808080', '#e43318', '#000000' ]
	n_groups = len(features)

	index = np.arange(n_groups)
	bar_width = 0.20
	opacity = 0.6

	#initialize averages and std. errors
	phn_feat_avg = []
	envs_avg = []
	pitch_avg = []
	full_avg = []

	phnfeat_se = []
	envs_se = []
	pitch_se = []
	full_se = []

	for i in np.arange(len(stim_type)): 
		if i == 0:
			phn_feat_avg.append(corrs_speech.mean(1)[:,i].mean(0))
			phnfeat_se.append(np.std(corrs_speech.mean(1)[:,i])/np.sqrt(corrs_speech.shape[0]))

			
			phn_feat_avg.append(corrs_speech_noise.mean(1)[:,i].mean(0))
			phnfeat_se.append(np.std(corrs_speech_noise.mean(1)[:,i])/np.sqrt(corrs_speech_noise.shape[0]))
			
			phn_feat_avg.append(corrs_noise.mean(1)[:,i].mean(0))
			phnfeat_se.append(np.std(corrs_noise.mean(1)[:,i])/np.sqrt(corrs_noise.shape[0]))
			

		elif i == 1:
			envs_avg.append(corrs_speech.mean(1)[:,i].mean(0))
			envs_se.append(np.std(corrs_speech.mean(1)[:,i])/np.sqrt(corrs_speech.shape[0]))
			
			envs_avg.append(corrs_speech_noise.mean(1)[:,i].mean(0))
			envs_se.append(np.std(corrs_speech_noise.mean(1)[:,i])/np.sqrt(corrs_speech_noise.shape[0]))
			
			envs_avg.append(corrs_noise.mean(1)[:,i].mean(0))
			envs_se.append(np.std(corrs_noise.mean(1)[:,i])/np.sqrt(corrs_noise.shape[0]))
			

		elif i == 2:
			pitch_avg.append(corrs_speech.mean(1)[:,i].mean(0))
			pitch_se.append(np.std(corrs_speech.mean(1)[:,i])/np.sqrt(corrs_speech.shape[0]))	

			pitch_avg.append(corrs_speech_noise.mean(1)[:,i].mean(0))
			pitch_se.append(np.std(corrs_speech_noise.mean(1)[:,i])/np.sqrt(corrs_speech_noise.shape[0]))
			
			pitch_avg.append(corrs_noise.mean(1)[:,i].mean(0))
			pitch_se.append(np.std(corrs_noise.mean(1)[:,i])/np.sqrt(corrs_noise.shape[0]))
			

		elif i == 3:
			full_avg.append(corrs_speech.mean(1)[:,i].mean(0))
			full_se.append(np.std(corrs_speech.mean(1)[:,i])/np.sqrt(corrs_speech.shape[0]))
			
			full_avg.append(corrs_speech_noise.mean(1)[:,i].mean(0))
			full_se.append(np.std(corrs_speech_noise.mean(1)[:,i])/np.sqrt(corrs_speech_noise.shape[0]))
			
			full_avg.append(corrs_noise.mean(1)[:,i].mean(0))
			full_se.append(np.std(corrs_noise.mean(1)[:,i])/np.sqrt(corrs_noise.shape[0]))
			
	#visualize in bar plot form
	rects1 = plt.bar(index - bar_width, phn_feat_avg, bar_width,  yerr = phnfeat_se, alpha=opacity, color='#cd1a1e', label='phnfeat')

	rects2 = plt.bar(index , pitch_avg, bar_width, yerr = pitch_se, alpha=opacity, color='#64a7bc',label='pitch')

	rects3 = plt.bar(index + bar_width, envs_avg, bar_width, yerr=envs_se, alpha=opacity, color='#808080',label='envs')

	rects4 = plt.bar(index + bar_width*2, full_avg, bar_width, yerr=full_se, alpha=opacity, color='#996666',label='full')
	plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='black')

	plt.xlabel('Features type')
	plt.ylabel('Average Correlation')
	plt.xticks(np.arange(len(features)), features)
	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

	if save_fig:
		plt.savefig('%s/auditory_segmentations_noSlidingCorr.pdf' %(save_dir))


def speech_nonspeech_tier_alignment_slidingCorr(grid_dir, grid_name, ntimes=17427, sfreq=128.0):
	'''
	Read in tier information from textgrid for test set trailer in order to get information about speech only, 
	speech with background sound, or background information only. 

	Parameters
	----------
	grid_dir : string 
		- directory where textgrid exists
	grid_name : string 
		- name of textgrid movie trailer 
	ntimes : int
		(default for Paddington, time of trailer in samples (128.0)	
	sfreq : float 
		(default sampling rate for downsampled EEG data)


	Returns
	-------
	clean_vs_noisy_speech : numpy array 
		- matrix of tier information where there is speech, speech with background noise, and background noise only 
			timepoint information, in samples 
	'''

	r = open('%s/%s'%(grid_dir, grid_name))
	grid = tg.TextGrid(r.read())
	tier_names = [t.nameid for t in grid.tiers]

	#print timing information in the Praat speech tier
	tier_num = [t for t, i in enumerate(tier_names) if i == 'speech'][0]
	speech_tier = [t for t in grid.tiers[tier_num].simple_transcript]
	speech_tier = np.array(speech_tier)

	#print timing information in the Praat music tier
	tier_num = [t for t, i in enumerate(tier_names) if i == 'music'][0]
	music_tier = [t for t in grid.tiers[tier_num].simple_transcript]
	music_tier = np.array(music_tier)

	#print timing information in the Praat background tier
	tier_num = [t for t, i in enumerate(tier_names) if i == 'background'][0]
	bg_tier = [t for t in grid.tiers[tier_num].simple_transcript]
	bg_tier = np.array(bg_tier)

	#speech_tier = np.array(speech_tier) #convert to numpy array 
	speech_condition_matrix = np.zeros((3,ntimes)) #initiailze matrix to append timing info based on auditory stimulus tier type

	for i in np.arange(speech_condition_matrix.shape[1]):
		t=i/sfreq
		for s in speech_tier: 
			onset_time = s[0].astype('float')
			offset_time = s[1].astype('float')
			label = s[2] 
			if (t >= onset_time) and (t <= offset_time) and (label=='SPEECH'):
				tsamp = np.int(t*sfreq)
				speech_condition_matrix[0,tsamp] = 1
		for s in music_tier: 
			onset_time = s[0].astype('float')
			offset_time = s[1].astype('float')
			label = s[2] 
			if (t >= onset_time) and (t <= offset_time) and (label=='MUSIC'):
				tsamp = np.int(t*sfreq)
				speech_condition_matrix[1,tsamp] = 1
		for s in bg_tier: 
			onset_time = s[0].astype('float')
			offset_time = s[1].astype('float')
			label = s[2] 
			if (t >= onset_time) and (t <= offset_time) and (label=='BG'):
				tsamp = np.int(t*sfreq)
				speech_condition_matrix[2,tsamp] = 1

	print('Plotting markers for where speech, music, and background sounds occur over the movie trailer specified: ')
	plt.imshow(speech_condition_matrix, aspect='auto', interpolation='nearest') 

	# Now loop through speech_condition_matrix to find when speech, music, background noise happens
	clean_vs_noisy_speech = np.zeros((ntimes,))
	for t in np.arange(speech_condition_matrix.shape[1]):
		if (speech_condition_matrix[0,t] == 1) and (speech_condition_matrix[2,t] == 1): # Clean speech with some background noise (either music or not)
			clean_vs_noisy_speech[t] = 2
		elif (speech_condition_matrix[0,t] == 1) and (speech_condition_matrix[2,t] == 0): # Clean speech, no background at all
			clean_vs_noisy_speech[t] = 1
		elif (speech_condition_matrix[0,t] == 0) and (speech_condition_matrix[2,t] == 1): # No speech, but there is background of some sort
			clean_vs_noisy_speech[t] = 3

	return clean_vs_noisy_speech

### SUPPLEMENTARY FIGURE FUNCTIONS ###
### Create function - for binned pitch vs. pitch alone vs full model corr comparison 
def binned10Pitch_vs_pitch(data_dir, subject_list, stimulus_class, full_model = False, pitch_only=True, save_fig=True):
	corrs = []
	for idx, s in enumerate(subject_list):
		if full_model:
			with h5py.File('%s/%s/%s_STRF_by_pitchenvsphnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model (old pitch)
				binned_full = fh['corrs_%s' %(stimulus_class.lower())][:]  
				p_val_binned_full =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_binned_pitches_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #full model, with binned pitch 
				pitch = h['corrs_%s' %(stimulus_class.lower())][:]
				p_vals_pitch = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'fullmodel_vs_fullbinnedpitch'

		if pitch_only:
			with h5py.File('%s/%s/%s_STRF_by_pitch_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #old pitch model only
				binned_full = fh['corrs_%s' %(stimulus_class.lower())][:]  
				p_val_binned_full =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_binned_pitch_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #binned pitch model only 
				pitch = h['corrs_%s' %(stimulus_class.lower())][:]
				p_vals_pitch = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'binned_vs_oldpitch'


		corrs.append([binned_full, pitch])
	points = np.hstack(corrs).T

	plt.plot(points[:,1], points[:,0], '.')
	plt.xlabel('pitch')
	plt.ylabel('binned pitch')


	if stimulus_class == 'TIMIT':
		plt.plot([-0.2, 0.6], [-0.2, 0.6], 'black', label='unity')

	else:
		plt.plot([-0.05, 0.2], [-0.05, 0.2], 'black', label='unity') 

	if save_fig:
		plt.savefig('/Users/%s/Box/MovieTrailersTask/Data/EEG/Revision_figs/%s_%s.pdf' %(user, stimulus_class, saving))

def gabor_scenecut_scatter(data_dir, subject_list, stimulus_class='MT', gabor_v_combined_model=True, sc_v_combined_model=False, save_fig=True):
	corrs = []
	for idx, s in enumerate(subject_list):
		if gabor_v_combined_model:
			with h5py.File('%s/%s/%s_STRF_by_scene_cut_gabor_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model
				sc_gabor_corrs = fh['corrs_%s' %(stimulus_class.lower())][:]  
				p_val_sc_gabor =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_gabor_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
				single_corrs = h['corrs_%s' %(stimulus_class.lower())][:]
				p_val = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'gabor_v_combined_model'
			label = 'gabor only'
			

		#corrs.append([sc_gabor_corrs, gabor_corrs])

		if sc_v_combined_model:
			with h5py.File('%s/%s/%s_STRF_by_scene_cut_gabor_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model
				sc_gabor_corrs = fh['corrs_%s' %(stimulus_class.lower())][:]  
				p_val_sc_gabor =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_scene_cut_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
				single_corrs = h['corrs_%s' %(stimulus_class.lower())][:]
				p_vals = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'sc_v_combined_model'
			label = 'scene cut'



		corrs.append([sc_gabor_corrs, single_corrs])
	
	points = np.hstack(corrs).T
	print('Printing wilcoxon results: ')
	print(pg.wilcoxon(points[:,1], points[:,0], tail='two-sided'))

	fig = plt.figure()
	ax = fig.subplots()
	#ax.set_aspect('equal')

	plt.plot(points[:,1], points[:,0], '.', color='#6a0dad') #plotting single visual feature
	plt.xlabel(label)
	plt.ylabel('scene cut + gabors') 
	


	plt.plot([0.0, 0.4], [0.0, 0.4], 'black', label='unity') 

	if save_fig:
		plt.savefig('/Users/%s/Box/MovieTrailersTask/Data/EEG/Revision_figs/%s_%s.pdf' %(user, stimulus_class, saving))



def convexHull_spec(stimulus_class, subject_list, data_dir, save_dir, save_fig=True): 
	
	fig = plt.figure(figsize=(15,9))
	ax = fig.subplots()
	ax.set_aspect('equal')
	if stimulus_class == 'TIMIT':
		plt.plot([-0.2, 0.6], [-0.2, 0.6], 'black', label='unity')
		
	elif stimulus_class == 'MT':
		plt.plot([-0.05, 0.2], [-0.05, 0.2], 'black', label='unity') 

	else:
		print('Undefined stimulus class')
	corrs = []
	corrs_sig = []
	corrs_nonsig = []
	for idx, s in enumerate(subject_list):
		with h5py.File('%s/%s/%s_STRF_by_pitchspecphnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model
			fullModel = fh['corrs_%s' %(stimulus_class.lower())][:]  
			p_vals_full =  fh['pvals_%s' %(stimulus_class.lower())][:]
			full_nonsig = (fullModel[np.where(p_vals_full[0] > 0.05)])
			full_sig = fullModel[np.where(p_vals_full[0] < 0.05)]
		with h5py.File('%s/%s/%s_STRF_by_spec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #envs model only 
			spec = h['corrs_%s' %(stimulus_class.lower())][:]
			p_vals_spec = h['pvals_%s' %(stimulus_class.lower())][:]
			spec_nonsig = (spec[np.where(p_vals_full[0] > 0.05)])
			spec_sig = (spec[np.where(p_vals_full[0] < 0.05)])
		with h5py.File('%s/%s/%s_STRF_by_pitch_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
			pitch = h['corrs_%s' %(stimulus_class.lower())][:]
			p_vals_pitch = h['pvals_%s' %(stimulus_class.lower())][:]
			pitch_nonsig = (pitch[np.where(p_vals_full[0] > 0.05)])
			pitch_sig = (pitch[np.where(p_vals_full[0] < 0.05)])
		with h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #phnfeat model only 
			phnfeat = h['corrs_%s' %(stimulus_class.lower())][:]
			p_vals_phnfeat = h['pvals_%s' %(stimulus_class.lower())][:]
			phnfeat_nonsig = (phnfeat[np.where(p_vals_full[0] > 0.05)])
			phnfeat_sig = (phnfeat[np.where(p_vals_full[0] < 0.05)])

		corrs.append([fullModel, spec, pitch, phnfeat])
		corrs_nonsig.append([full_nonsig, spec_nonsig, pitch_nonsig, phnfeat_nonsig])
		corrs_sig.append([full_sig, spec_sig, pitch_sig, phnfeat_sig])

	points = np.hstack(corrs).T
	points_sig = np.hstack(corrs_sig).T
	points_nonsig = np.hstack(corrs_nonsig).T

	hull1=ConvexHull(np.vstack((points[:,0] , points[:,1])).T)
	hullv1 = hull1.vertices.copy()
	hullv1 = np.append(hullv1, hullv1[0])
	print(hullv1)

	hull2=ConvexHull(np.vstack((points[:,0] , points[:,2])).T)
	hullv2 = hull2.vertices.copy()
	hullv2 = np.append(hullv2, hullv2[0])
	print(hullv2)


	hull3=ConvexHull(np.vstack((points[:,0] , points[:,3])).T)
	hullv3 = hull3.vertices.copy()
	hullv3 = np.append(hullv3, hullv3[0])
	print(hullv3)


	#fill between: 
	plt.fill(points[hullv3,0], points[hullv3,3], facecolor='#cd1a1e', alpha=0.4, zorder=1, label='full (spec) vs. phnfeat') #phn feat 
	plt.fill(points[hullv1,0], points[hullv1,1], facecolor='#78AB46', alpha=0.4, zorder=2, label='full (spec) vs. spec') #spec
	plt.fill(points[hullv2,0], points[hullv2,2], facecolor='#64a7bc', alpha=0.4, zorder=3, label='full (spec) vs. pitch') #pitch
	# plt.fill(points[hullv4,0], points[hullv4,4], facecolor='#4B8B3B', alpha=0.4, zorder=4, label='full vs. binned_pitch')

	plt.plot(points_sig[:,0], points_sig[:,3], '.', color='#cd1a1e',alpha=0.8) #phnfeat
	plt.plot(points_sig[:,0], points_sig[:,1], '.', color='#78AB46', alpha=0.8) #spec
	plt.plot(points_sig[:,0], points_sig[:,2], '.', color='#64a7bc', alpha=0.8) #pitch
	# plt.plot(points_sig[:,0], points_sig[:,4], '.', color='#4B8B3B', alpha=0.8) #binned pitch 

	plt.plot(points_nonsig[:,0], points_nonsig[:,3], '.', color='#CDC1C5',alpha=0.7)
	plt.plot(points_nonsig[:,0], points_nonsig[:,1], '.', color='#CDC1C5', alpha=0.7)
	plt.plot(points_nonsig[:,0], points_nonsig[:,2], '.', color='#CDC1C5', alpha=0.7)
	# plt.plot(points_nonsig[:,0], points_nonsig[:,4], '.', color='#CDC1C5', alpha=0.7) #binned pitch non sig
	
	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
	plt.xlabel('Full model correlation values (r)')
	plt.ylabel('Individual model correlation values (r)')
	plt.title('Convex-Hull for %s (Feature distribution across subjects)' %(stimulus_class))
	plt.axis('square')

	if save_fig:
		plt.savefig('%s/%s_ConvexHull_specFull.pdf' %(save_dir, stimulus_class))

def var_partition_spec(stimulus_class, subject_list, data_dir, opacity=0.4):
	"""
	stimulus_class : string 
		- MT or TIMIT 
	subject_list : list
		- inputted from create_subject_list function 
	data_dir : string 
		-change this to match where all inidivdual .h5 files are 
	opacity : float
		- transparency for all colors in bar plot 
		(default: 0.4)
	"""
	
	variance_partition_spec = []

	for idx, s in enumerate(subject_list):
		with h5py.File('%s/%s/%s_STRF_by_pitchspecphnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #full model
			corr_full = h['corrs_%s' %(stimulus_class.lower())][:] 
			p_val_full = h['pvals_%s' %(stimulus_class.lower())][:]  
		with h5py.File('%s/%s/%s_STRF_by_pitchspec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate phnfeat model var
			corr_pitchspec=h['corrs_%s' %(stimulus_class.lower())][:]
			#p_val_envspitch = h['pvals_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_pitchphnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate envs model var:
			corr_pitchphnfeat=h['corrs_%s' %(stimulus_class.lower())][:]
			#p_val_pitchphnfeat = h['pvals_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_phnfeatspec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate pitch model var
			corr_phnfeatspec=h['corrs_%s' %(stimulus_class.lower())][:]
			#p_val_envsphnfeat = h['pvals_%s'%(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_spec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #envs model only 
			corr_spec = h['corrs_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_pitch_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
			corr_pitch = h['corrs_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #phnfeat model only 
			corr_phnfeat = h['corrs_%s' %(stimulus_class.lower())][:]

		sig_corrs_full = ([np.where (p_val_full[0] < 0.05)])
		full_model = corr_full[sig_corrs_full] #sig elec corrs for full model 
		spec_U_pitch = corr_pitchspec[sig_corrs_full]
		pitch_U_phnfeat = corr_pitchphnfeat[sig_corrs_full]
		spec_U_phnfeat = corr_phnfeatspec[sig_corrs_full]         
		spec_only = corr_spec[sig_corrs_full]
		pitch_only = corr_pitch[sig_corrs_full]
		phnfeat_only = corr_phnfeat[sig_corrs_full]
		
		variance_partition_spec.append(np.array([spec_only, phnfeat_only, pitch_only, 
											spec_U_phnfeat, spec_U_pitch, pitch_U_phnfeat, full_model]))

	return variance_partition_spec

def create_vennDiagram_spec(subject_list, stimulus_class, variance_partition_spec, save_dir, save_fig=True):
	"""
	subject_list : list
		- inputted from create_subject_list function 
	stimulus_class : string 
		MT or TIMIT 
	variance_partition : return function from var_partition function 
	save_fig : bool 
		- outputs figure/plot when True
		(default : True)
	save_dir : string 
		-change this to match where to save the plot for each condition in PDF format 
	"""

	rsq_corr = lambda c: (c ** 2) * np.sign(c) # r -> r^2 

	venn_diagram_spec = []
	for idx, s in enumerate(subject_list):
		corr_est = np.squeeze(variance_partition[idx])

		# estimate biases, original partitions, and fixed partitions using r^2
		corr_biases, corr_orig_parts, corr_fixed_parts = correct_rsqs(rsq_corr(np.array(corr_est)), neg_only=False)
		venn_diagram.append(corr_fixed_parts.mean(1))
	#print(venn_diagram)
	venn_avg = np.array(venn_diagram_spec).mean(0)

	# [envs_only, phnfeat_only, pitch_only, envs_U_phnfeat, envs_U_pitch, pitch_U_phnfeat, full_model]

	v=venn3(subsets = (venn_avg[1], venn_avg[0], venn_avg[3], venn_avg[2], 
					  venn_avg[5], venn_avg[4], venn_avg[6]),
			set_labels = ('Phonological Features', 'Spectrogram', 'Pitch'), alpha=0.4)

	v.get_patch_by_id('100').set_color('#cd1a1e') #phnfeat only
	v.get_patch_by_id('001').set_color('#64a7bc') #pitch only
	v.get_patch_by_id('010').set_color('#78AB46') #spec only
	v.get_patch_by_id('101').set_color('#946774') #phnfeat + pitch intersection
	v.get_patch_by_id('110').set_color('#9e6a34') #phnfeat + spec intersection
	v.get_patch_by_id('011').set_color('#6ea981') #pitch + spec intersection
	v.get_patch_by_id('111').set_color('#8e7960') #all comb

	# #remove numbers from the venn diagram for each section 
	for idx, subset in enumerate(v.subset_labels):
		v.subset_labels[idx].set_visible(False)
	
	if save_fig:
			plt.savefig('%s/%s_vennDiagram_spec.pdf' %(save_dir, stimulus_class))
	return venn_diagram_spec

def unique_variance_Barplot_spec(stimulus_class, subject_list, venn_diagram_spec, save_dir, opacity=0.4, save_fig=True):

	n_groups = len(subject_list)

	plt.figure()
	index = np.arange(n_groups)
	bar_width = 0.25

	unique_var_pitch = []
	unique_var_phnfeat = []
	unique_var_spec = []

	for idx, s in enumerate(subject_list):
		unique_var_pitch.append(venn_diagram[idx][2]) #spec_U_phnfeat
		unique_var_phnfeat.append(venn_diagram[idx][1]) #spec_U_pitch
		unique_var_spec.append(venn_diagram[idx][0]) #pitch_U_phnfeat

	df = pd.DataFrame(list(zip(unique_var_pitch, unique_var_phnfeat, unique_var_spec, np.array(unique_var_pitch) + np.array(unique_var_phnfeat) + np.array(unique_var_spec))), 
		columns=['pitch', 'phnfeat', 'spec', 'sum']) #create dataframe from list
	df.index = subject_list 

	my_colors = ['#64a7bc', '#cd1a1e', '#78AB46']
	df.sort_values(by='sum', ascending=False)[['pitch', 'phnfeat', 'spec']].plot(kind='bar', stacked=True, color=my_colors, alpha=opacity, width=0.5)


	plt.xlabel('Participant number')
	plt.ylabel('Unique variance')
	plt.title('Unique feature variance (%s)' %(stimulus_class))

	#plt.xticks(np.arange(len(subject_list)), np.array(subject_list), rotation=90)

	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
	
	if save_fig:
		plt.savefig('%s/%s_uniqueVar_w_spec.pdf' %(save_dir, stimulus_class))


	
def weights_corr_coef(subject_list, data_dir, nchans = np.arange(64), fs = 128.0, ndelays = 77, nfeats=25, delay_min=0.0, delay_max=0.6): 
	## NEW function to plot weights
	stimulus_classes = ['TIMIT', 'MT']
	for stimulus_class in stimulus_classes:
		if stimulus_class == 'TIMIT':
			wts_matrix_timit = np.zeros((nfeats*ndelays, len(nchans), len(subject_list))) #(ndelays*nfeats, nchans, nsubj) or (2002 x 64 x 16)
		elif stimulus_class == 'MT':
			wts_matrix_mt = np.zeros((nfeats*ndelays, len(nchans), len(subject_list))) #(ndelays*nfeats, nchans, nsubj)
		else:
			print('Undefined input')

		for idx, s in enumerate(subject_list):
			with h5py.File('%s/%s/%s_STRF_by_binned_pitches_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model
				p_vals =  fh['pvals_%s' %(stimulus_class.lower())][:]
				wts = fh['wts_%s' %(stimulus_class.lower())][:]

				if stimulus_class == 'TIMIT':
					wts_matrix_timit[:,:, idx]=wts
				elif stimulus_class == 'MT':
					wts_matrix_mt[:,:, idx]=wts
				else:
					print('Undefined input')

	wts_corrs_list = np.zeros((len(nchans), len(subject_list)))
	for ii, t in enumerate(subject_list):
		for channel in np.arange(64):
			timit = wts_matrix_timit[:,channel,ii] #average timit wts across channels

			mt = wts_matrix_mt[:,channel,ii] #average mt wts across channels
			x = np.corrcoef(timit, mt)
			wts_corrs_list[channel,ii] = x[0,1] #setting every element of corr matrix to channel and subj

	pitch_labels = np.arange(10).astype(str)
	feat_labels = ['sonorant','obstruent','voiced','back','front','low','high','dorsal', 
	'coronal','labial','syllabic','plosive','fricative','nasal', 'env']

	feat_labels = feat_labels + pitch_labels.tolist()


	nfeats = len(feat_labels)
	ds = load_ICA_data(subject_list[0], data_dir, file='post_ICA', preload=True)

	if 'STI 014' in ds.info['ch_names']:
		ds.drop_channels(['vEOG', 'hEOG', 'STI 014'])
	else:
		ds.drop_channels(['vEOG', 'hEOG'])
		
	chnames = ds.info['ch_names']
	new_order = np.array(chnames).argsort()

	print(wts_corrs_list.argmax())
	y = wts_corrs_list.argmax()
	[time, ch, nsubj] = np.unravel_index(y, (wts_matrix_mt.shape))

	topo_input = wts_corrs_list.mean(1)


	#figure for movie trailer weights correlation 
	plt.figure(); plt.imshow(wts_matrix_mt[:,ch,nsubj].reshape(ndelays, nfeats).T, aspect='auto', cmap=cm.RdBu_r, vmin=-np.abs(wts_matrix_mt[:,ch,nsubj]).max(), vmax=np.abs(wts_matrix_mt[:,ch,nsubj]).max()) #input correctly?
	plt.colorbar()
	t = np.linspace(delay_min, delay_max, ndelays)
	#plt.gca().set_xticks([0, (len(ndelays)-1)/2, ndelays-1])
	plt.gca().set_xticks([0, (ndelays-1)/2, ndelays-1])
	plt.gca().set_xticklabels([t[0], t[np.int((ndelays-1)/2)], t[ndelays-1]])
	plt.xlabel('Time (seconds)')
	plt.gca().set_yticks(np.arange(len(feat_labels)))
	plt.gca().set_yticklabels(feat_labels)	
	plt.title('Trailers weight correlation')
	plt.savefig('%s/mt_wt_corrs.pdf' %(save_dir))

	#figure for TIMIT weights correlation
	plt.figure(); plt.imshow(wts_matrix_timit[:,ch,nsubj].reshape(ndelays, nfeats).T, aspect='auto', cmap=cm.RdBu, vmin=-np.abs(wts_matrix_timit[:,ch,nsubj]).max(), vmax=np.abs(wts_matrix_timit[:,ch,nsubj]).max()) #input correctly?
	plt.colorbar()
	t = np.linspace(delay_min, delay_max, ndelays)
	plt.gca().set_xticks([0, (ndelays-1)/2, ndelays-1])
	plt.gca().set_xticklabels([t[0], t[np.int((ndelays-1)/2)], t[ndelays-1]])
	plt.xlabel('Time (seconds)')
	plt.gca().set_yticks(np.arange(len(feat_labels)))
	plt.gca().set_yticklabels(feat_labels)	#subjects?
	plt.title('TIMIT weight correlation')
	plt.savefig('%s/timit_wt_corrs.pdf' %(save_dir))

	#create and save topo map
	plt.figure() 
	info = mne.create_info(ch_names=ds.info['ch_names'][:64], sfreq=ds.info['sfreq'], ch_types=64*['eeg'])
	raw2 = mne.io.RawArray(np.zeros((64,10)), info)
	montage = mne.channels.read_montage('%s/montage/AP-128.bvef' %(data_dir), unit='mm')
	raw2.set_montage(montage)
	mne.viz.plot_topomap(topo_input, raw2.info, vmin=0, vmax=topo_input.max())
	plt.savefig('%s/avg_weights_Subjtopo.pdf' %(save_dir))

	#create and save color bar 
	fig = plt.figure()
	ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
	cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cm.Reds, norm=mpl.colors.Normalize(topo_input.min(), topo_input.max()))
	plt.savefig('%s/colorbar_for_topo.pdf' %(save_dir))


def scene_cut_feature(h5_dir, textgrid_dir, fs=128.0):
	'''
	Uses the textgrid scene cuts for movie trailers and creates a binary matrix for 
	where a scene cut takes place. 

	Appends instances of scene cuts in samples (fs=128.0) to fullEEGmatrix which contains 
	stim and resp for all subjects with corresponding auditory (and visual for MTs) features
	'''
	with h5py.File('%s/fullEEGmatrix.hf5'%(h5_dir), 'a') as g:

		for name in glob.glob('%s/*.TextGrid' %(textgrid_dir)):
			print(name)
			r = open(name)
			grid = tg.TextGrid(r.read())
			tier_names = [t.nameid for t in grid.tiers]
			for r, i in enumerate(tier_names):
				if i == 'Scene cut':

					print(r)

			scene_tier = [t for t in grid.tiers[r].simple_transcript]
			scene_tier = np.array(scene_tier)

			scene_onset = scene_tier[:,0].astype(float)*fs
			scene_onset = scene_onset.astype(int)
			scene_offset = scene_tier[:,1].astype(float)*fs
			scene_offset = scene_offset.astype(int)

			name = name.replace(textgrid_dir, '')
			name = name.replace('_SC.TextGrid', '.wav')
			name = name.replace('_corrected', '')
			print(name)

			#create matrix length of trailer and append scene onset as 1 
			matrix_values = np.zeros((scene_offset[-1],))
			for i in scene_onset:
				matrix_values[i] = 1 
			print(matrix_values.shape)
			matrix_dims = np.expand_dims(matrix_values, axis=0)

			#append to h5 file /stim/%s/scene_cut as binary matrix 
			g.create_dataset('MovieTrailers/%s/stim/scene_cut'%(name), data=np.array(matrix_dims, dtype=float))

def stim_time(subject, stimulus_class):
	'''
	Read through aligned audio files from EEG experiment for each subject to get the onset, offset, and total duration of stimulus played 
	for either TIMIT or movie trailers 

	Parameters
	----------
	subject : string 
	stimulus_class : string 
		- TIMIT or MovieTrailers as the input

	Returns
	-------
	audio_len : float 
		- length of total experiment time
	total_stim_time : float 
		- length of total auditory stimulus time heard per stimulus set 
	stim_silence : 
		- total amount of silence in task for subject

	'''

	if stimulus_class == 'TIMIT':
		print(stimulus_class)
		audio_txt = '%s/%s/audio/%s_%s_all_events.txt' %(data_dir, subject, subject, stimulus_class)

	else:
		print(stimulus_class)

		audio_txt = '%s/%s/audio/%s_%s_events.txt' %(data_dir, subject, subject, stimulus_class)
	read_audio_txt = np.loadtxt(audio_txt, dtype=np.float, usecols=([0,1]))
	onset = read_audio_txt[:,0:]
	offset = read_audio_txt[:,1:]
	
	fs, wave = scipy.io.wavfile.read('%s/%s/audio/%s_16kHz.wav' %(data_dir, subject, subject))
	print ('Data:', wave)
	print ('Sampling rate:', fs)
	print ('Audio length:', wave.size/fs, 'seconds')

	audio_len = wave.size/fs
	total_stim_time = np.sum(offset-onset)
	stim_silence = audio_len - total_stim_time
	

	return audio_len, total_stim_time, stim_silence


#loop through all subject + stimulus_class
def stimulus_time_mat(audio_len, total_stim_time, stim_silence, subject_list):
	'''
	create time matrix for each stimuli with the total amount of sound heard from task, silence, and full task time 
	User audio_len, total_stim_time, and stim_silence from the stim_time function 

	'''

	stim_time_array = np.zeros((len(subject_list), 4))
	for idx, subj in enumerate(subject_list):
		print('****************************')
		print(subj)
		print('****************************')
		audio_len, timit_total_stim_time, stim_silence = stim_time(subj, 'TIMIT')
		audio_len, mt_total_stim_time, stim_silence = stim_time(subj, 'MovieTrailers')

		stim_time_array[idx] = [audio_len, stim_silence, timit_total_stim_time, mt_total_stim_time]

	timit = []
	mt = []
	total_time = []
	silence = []
	for i in stim_time_array:
		total_time.append(i[0])
		silence.append(i[1])
		timit.append(i[0]-i[2])
		mt.append(i[0]-i[3])


	return stim_time_array, timit, mt, total_time


def covariance_feature_mat(stimulus_class, data_dir, save_dir, subject='MT0002'):
	'''
	Correlation matrix plot to identify how correlate all of the auditory features are to one another 
	Function also creates a scatter plot to compare the correlations between auditory features for both TIMIT and movie trailers 

	Parameters
	----------
	stimulus_class : string 
		- 'TIMIT' or 'MovieTrailers'
	data_dir : string 
		- path to data
	save_dir : string 
		- path to save figures 
	subject : string 
		(default : MT0002) - can be any participant because stimulus dictionary is the same across all subjects 

	Returns
	-------
	1) Confusion matrix of stimulus feature correlations 
	2) Scatter plot comparing correlation between auditory features for both stimulus sets 

	'''

	#load all auditory features from stimulus played 
	resp_dict, stim_dict, freqs = loadEEGh5(subject, stimulus_class, data_dir,
		eeg_epochs=True, resp_mean = True, binarymat=False, binaryfeatmat = True, envelope=True, pitch=False, gabor_pc10=False, 
		spectrogram=True, binned_pitches=True, spectrogram_scaled=False, scene_cut=False)

	all_stimuli = [k for k in (stim_dict).keys() if len(resp_dict[k]) > 0]
	train_inds = np.zeros((len(all_stimuli),), dtype=np.bool)

	for i in np.arange(len(all_stimuli)):
		train_inds[i] = True

	print(sum(train_inds))

	train_inds = np.where(train_inds==True)[0]
	print(train_inds)

	delay_min = 0.0
	delay_max = 0.0

	wt_pad = 0.0
	fs = 128.0

	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) #create array to pass time delays in
	delays = [0]
	print("Delays:", delays)

	current_stim_list_train = np.array([all_stimuli[r][0] for r in train_inds])
	test_set = []

	training_set = np.setdiff1d(all_stimuli, test_set)

	tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
	tStim = make_delayed(tStim_temp, delays)

	print(tStim.shape)

	#covariance:
	spec_labels = np.arange(15).astype(str)
	pitch_labels = np.arange(10).astype(str)
	feat_labels = ['sonorant','obstruent','voiced','back','front','low','high','dorsal', 
	'coronal','labial','syllabic','plosive','fricative','nasal', 'env']

	new_feats = feat_labels + spec_labels.tolist() + pitch_labels.tolist()

	nfeats = len(new_feats)
	correlate_mat = np.corrcoef(tStim.T)
	plt.imshow(correlate_mat, aspect='auto', cmap=cm.RdBu_r, vmin=-abs(correlate_mat).max(), vmax=abs(correlate_mat).max())
	plt.gca().set_xticks(np.arange(correlate_mat.shape[0]))
	plt.gca().set_xticklabels(new_feats, rotation=90)

	plt.gca().set_yticks(np.arange(correlate_mat.shape[0]))
	plt.gca().set_yticklabels(new_feats)
	plt.colorbar()
	plt.title('Feature Covariances - %s' %stimulus_class)
	plt.savefig('%s/feature_covarMatrix_%s.pdf' %(save_dir, stimulus_class))
	plt.show()

	#find average corr coefficient between auditory features to plot:
	phn_v_envs = correlate_mat[:13,14].mean() 
	phn_v_pitch = correlate_mat[:13,30:].mean()
	phn_v_spec = correlate_mat[:13,16:29].mean()
	envs_v_pitch = correlate_mat[14,30:].mean()
	envs_v_spec = correlate_mat[14,16:29].mean()
	pitch_v_spec = correlate_mat[30:,16:29].mean()
	covar_avg = [phn_v_envs, phn_v_pitch, phn_v_spec, envs_v_pitch, envs_v_spec, pitch_v_spec]

	labels = ['phns_vs_envs', 'phns_vs_pitch', 'phns_vs_spec', 'envs_vs_pitch', 'envs_vs_spec', 'pitch_vs_spec']

	if stimulus_class == 'TIMIT':
		color = 'red'
	else:
		color = 'blue'

	for idx, i in enumerate(covar_avg):
		plt.plot(i, idx, '.', color=color)
	plt.gca().set_yticks(np.arange(len(covar_avg)))
	plt.gca().set_yticklabels(labels)
	plt.savefig('%s/feature_covarMatrix_corrs.pdf' %(save_dir))

	return correlate_mat

def binned10Pitch_vs_pitch(data_dir, subject_list, stimulus_class, full_model = False, pitch_only=True, save_fig=True):
	corrs = []
	for idx, s in enumerate(subject_list):
		if full_model:
			with h5py.File('%s/%s/%s_STRF_by_pitchenvsphnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model (old pitch)
				binned_full = fh['corrs_%s_norm' %(stimulus_class.lower())][:]  
				p_val_binned_full =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_binned_pitches_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #full model, with binned pitch 
				pitch = h['corrs_%s' %(stimulus_class.lower())][:]
				p_vals_pitch = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'fullmodel_vs_fullbinnedpitch'

		if pitch_only:
			with h5py.File('%s/%s/%s_STRF_by_pitch_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #old pitch model only
				binned_full = fh['corrs_%s_norm' %(stimulus_class.lower())][:]  
				p_val_binned_full =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_binned_pitch_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #binned pitch model only 
				pitch = h['corrs_%s_norm' %(stimulus_class.lower())][:]
				p_vals_pitch = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'binned_vs_oldpitch'


		corrs.append([binned_full, pitch])
	points = np.hstack(corrs).T

	plt.figure()
	plt.plot(points[:,1], points[:,0], '.')
	plt.xlabel('pitch')
	plt.ylabel('binned pitch')


	if stimulus_class == 'TIMIT':
		plt.plot([-0.75, 1.0], [-0.75, 1.0], 'black', label='unity')

	else:
		plt.plot([-0.2, 0.4], [-0.2, 0.4], 'black', label='unity') 

	if save_fig:
		plt.savefig('/Users/%s/Box/MovieTrailersTask/Data/EEG/Revision_figs/%s_%s.pdf' %(user, stimulus_class, saving))

def gabor_scenecut_scatter(data_dir, subject_list, stimulus_class='MT', gabor_v_combined_model=True, sc_v_combined_model=False, save_fig=True):
	corrs = []
	for idx, s in enumerate(subject_list):
		f = h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r')
		cc_max=f['/cc_max_%s' %(stimulus_class.lower())][:]
		if gabor_v_combined_model:
			with h5py.File('%s/%s/%s_STRF_by_scene_cut_gabor_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model
				sc_gabor_corrs = fh['corrs_%s' %(stimulus_class.lower())][:]/cc_max
				p_val_sc_gabor =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_gabor_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
				single_corrs = h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
				p_val = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'gabor_v_combined_model'
			label = 'gabor only'
			

		#corrs.append([sc_gabor_corrs, gabor_corrs])

		if sc_v_combined_model:
			with h5py.File('%s/%s/%s_STRF_by_scene_cut_gabor_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model
				sc_gabor_corrs = fh['corrs_%s' %(stimulus_class.lower())][:]/cc_max 
				p_val_sc_gabor =  fh['pvals_%s' %(stimulus_class.lower())][:]

			with h5py.File('%s/%s/%s_STRF_by_scene_cut_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
				single_corrs = h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
				p_vals = h['pvals_%s' %(stimulus_class.lower())][:]
			saving = 'sc_v_combined_model'
			label = 'scene cut'



		corrs.append([sc_gabor_corrs, single_corrs])
	
	points = np.hstack(corrs).T
	print('Printing wilcoxon results: ')
	print(pg.wilcoxon(points[:,1], points[:,0], tail='two-sided'))

	fig = plt.figure()
	ax = fig.subplots()
	#ax.set_aspect('equal')

	plt.plot(points[:,1], points[:,0], '.', color='#6a0dad') #plotting single visual feature
	plt.xlabel(label)
	plt.ylabel('scene cut + gabors') 
	


	plt.plot([-0.2, 0.8], [-0.2, 0.8], 'black', label='unity') 

	if save_fig:
		plt.savefig('/Users/%s/Box/MovieTrailersTask/Data/EEG/Revision_figs/%s_%s.pdf' %(user, stimulus_class, saving))



def convexHull_spec(stimulus_class, subject_list, data_dir, save_dir, save_fig=True): 
	
	fig = plt.figure(figsize=(15,9))
	ax = fig.subplots()
	ax.set_aspect('equal')
	if stimulus_class == 'TIMIT':
		plt.plot([-0.2, 1.00], [-0.2, 1.00], 'black', label='unity')
		
	elif stimulus_class == 'MT':
		plt.plot([-0.1, 0.6], [-0.1, 0.6], 'black', label='unity') 

	else:
		print('Undefined stimulus class')

# plt.plot(corrs_mt[np.where (pval_mt[0] > 0.05)], np.take(corr_mt_from_timit, np.asarray(np.where (pval_mt[0] > 0.05)))[0], '.', color='#CDC1C5')
# plt.plot(corrs_mt[np.where (pval_mt[0] < 0.05)], np.take(corr_mt_from_timit, np.asarray(np.where (pval_mt[0] < 0.05))[0]), '.')
	corrs = []
	corrs_sig = []
	#corrs_nonsig = []
	for idx, s in enumerate(subject_list):
		f = h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r')
		cc_max=f['/cc_max_%s' %(stimulus_class.lower())][:]
		print(cc_max)
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_spec_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as fh: #full model
			fullModel = fh['corrs_%s' %(stimulus_class.lower())][:]/cc_max
			p_vals_full =  fh['pvals_%s' %(stimulus_class.lower())][:]
			full_sig = fullModel[np.where(p_vals_full[0] < 0.05)]
		with h5py.File('%s/%s/%s_STRF_by_spec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #envs model only 
			spec = h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
			p_vals_spec = h['pvals_%s' %(stimulus_class.lower())][:]
			spec_sig = (spec[np.where(p_vals_full[0] < 0.05)])
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
			pitch = h['corrs_%s_norm' %(stimulus_class.lower())][:]
			
			# p_vals_pitch = h['pvals_%s' %(stimulus_class.lower())][:]
			# pitch_nonsig = (pitch[np.where(p_vals_full[0] > 0.05)])
			pitch_sig = (pitch[np.where(p_vals_full[0] < 0.05)])
		with h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #phnfeat model only 
			phnfeat = h['corrs_%s_norm' %(stimulus_class.lower())][:]
			
			# p_vals_phnfeat = h['pvals_%s' %(stimulus_class.lower())][:]
			# phnfeat_nonsig = (phnfeat[np.where(p_vals_full[0] > 0.05)])
			phnfeat_sig = (phnfeat[np.where(p_vals_full[0] < 0.05)])

		corrs.append([fullModel, spec, pitch, phnfeat])
		#corrs_nonsig.append([full_nonsig, spec_nonsig, pitch_nonsig, phnfeat_nonsig])
		corrs_sig.append([full_sig, spec_sig, pitch_sig, phnfeat_sig])

	points = np.hstack(corrs).T
	points_sig = np.hstack(corrs_sig).T
	#points_nonsig = np.hstack(corrs_nonsig).T

	hull1=ConvexHull(np.vstack((points_sig[:,0] , points_sig[:,1])).T)
	hullv1 = hull1.vertices.copy()
	hullv1 = np.append(hullv1, hullv1[0])
	print(hullv1)

	hull2=ConvexHull(np.vstack((points_sig[:,0] , points_sig[:,2])).T)
	hullv2 = hull2.vertices.copy()
	hullv2 = np.append(hullv2, hullv2[0])
	print(hullv2)


	hull3=ConvexHull(np.vstack((points_sig[:,0] , points_sig[:,3])).T)
	hullv3 = hull3.vertices.copy()
	hullv3 = np.append(hullv3, hullv3[0])
	print(hullv3)


	#fill between: 
	plt.fill(points_sig[hullv3,0], points_sig[hullv3,3], facecolor='#cd1a1e', alpha=0.4, zorder=1, label='full (spec) vs. phnfeat') #phn feat 
	plt.fill(points_sig[hullv1,0], points_sig[hullv1,1], facecolor='#78AB46', alpha=0.4, zorder=2, label='full (spec) vs. spec') #spec
	plt.fill(points_sig[hullv2,0], points_sig[hullv2,2], facecolor='#64a7bc', alpha=0.4, zorder=3, label='full (spec) vs. pitch') #pitch
	# plt.fill(points[hullv4,0], points[hullv4,4], facecolor='#4B8B3B', alpha=0.4, zorder=4, label='full vs. binned_pitch')

	plt.plot(points_sig[:,0], points_sig[:,3], '.', color='#cd1a1e',alpha=0.8) #phnfeat
	plt.plot(points_sig[:,0], points_sig[:,1], '.', color='#78AB46', alpha=0.8) #spec
	plt.plot(points_sig[:,0], points_sig[:,2], '.', color='#64a7bc', alpha=0.8) #pitch
	# plt.plot(points_sig[:,0], points_sig[:,4], '.', color='#4B8B3B', alpha=0.8) #binned pitch 

	# plt.plot(points_nonsig[:,0], points_nonsig[:,4], '.', color='#CDC1C5', alpha=0.7) #binned pitch non sig
	
	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
	plt.xlabel('Full model correlation values (r)')
	plt.ylabel('Individual model correlation values (r)')
	plt.title('Convex-Hull for %s (Feature distribution across subjects)' %(stimulus_class))
	plt.axis('square')

	if save_fig:
		plt.savefig('%s/%s_ConvexHull_specFull.pdf' %(save_dir, stimulus_class))

def var_partition_spec(stimulus_class, subject_list, data_dir, opacity=0.4):
	"""
	stimulus_class : string 
		- MT or TIMIT 
	subject_list : list
		- inputted from create_subject_list function 
	data_dir : string 
		-change this to match where all inidivdual .h5 files are 
	opacity : float
		- transparency for all colors in bar plot 
		(default: 0.4)
	"""
	
	variance_partition_spec = []

	for idx, s in enumerate(subject_list):
		f = h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r')
		cc_max=f['/cc_max_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_spec_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #full model
			corr_full = h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
			p_val_full = h['pvals_%s' %(stimulus_class.lower())][:]  
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_spec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate phnfeat model var
			corr_pitchspec=h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
			#p_val_envspitch = h['pvals_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate envs model var:
			corr_pitchphnfeat=h['corrs_%s' %(stimulus_class.lower())][:]
			#p_val_pitchphnfeat = h['pvals_%s' %(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_phnfeatspec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #calculate pitch model var
			corr_phnfeatspec=h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
			#p_val_envsphnfeat = h['pvals_%s'%(stimulus_class.lower())][:]
		with h5py.File('%s/%s/%s_STRF_by_spec_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #envs model only 
			corr_spec = h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
		with h5py.File('%s/%s/%s_STRF_by_binned_pitch_only_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #pitch model only 
			corr_pitch = h['corrs_%s' %(stimulus_class.lower())][:]/cc_max
		with h5py.File('%s/%s/%s_STRF_by_phnfeat_%s.hf5'%(data_dir, s, s, stimulus_class), 'r') as h: #phnfeat model only 
			corr_phnfeat = h['corrs_%s' %(stimulus_class.lower())][:]/cc_max

		sig_corrs_full = ([np.where (p_val_full[0] < 0.05)])
		full_model = corr_full[sig_corrs_full] #sig elec corrs for full model 
		spec_U_pitch = corr_pitchspec[sig_corrs_full]
		pitch_U_phnfeat = corr_pitchphnfeat[sig_corrs_full]
		spec_U_phnfeat = corr_phnfeatspec[sig_corrs_full]         
		spec_only = corr_spec[sig_corrs_full]
		pitch_only = corr_pitch[sig_corrs_full]
		phnfeat_only = corr_phnfeat[sig_corrs_full]
		
		variance_partition_spec.append(np.array([spec_only, phnfeat_only, pitch_only, 
											spec_U_phnfeat, spec_U_pitch, pitch_U_phnfeat, full_model]))

	return variance_partition_spec

def create_vennDiagram_spec(subject_list, stimulus_class, variance_partition_spec, save_dir, save_fig=True):
	"""
	subject_list : list
		- inputted from create_subject_list function 
	stimulus_class : string 
		MT or TIMIT 
	variance_partition : return function from var_partition function 
	save_fig : bool 
		- outputs figure/plot when True
		(default : True)
	save_dir : string 
		-change this to match where to save the plot for each condition in PDF format 
	"""

	rsq_corr = lambda c: (c ** 2) * np.sign(c) # r -> r^2 

	venn_diagram = []
	for idx, s in enumerate(subject_list):
		corr_est = np.squeeze(variance_partition_spec[idx])

		# estimate biases, original partitions, and fixed partitions using r^2
		corr_biases, corr_orig_parts, corr_fixed_parts = correct_rsqs(rsq_corr(np.array(corr_est)), neg_only=False)
		venn_diagram.append(corr_fixed_parts.mean(1))
	#print(venn_diagram)
	venn_avg = np.array(venn_diagram).mean(0)

	# [envs_only, phnfeat_only, pitch_only, envs_U_phnfeat, envs_U_pitch, pitch_U_phnfeat, full_model]

	v=venn3(subsets = (venn_avg[1], venn_avg[0], venn_avg[3], venn_avg[2], 
					  venn_avg[5], venn_avg[4], venn_avg[6]),
			set_labels = ('Phonological Features', 'Spectrogram', 'Pitch'), alpha=0.4)

	v.get_patch_by_id('100').set_color('#cd1a1e') #phnfeat only
	v.get_patch_by_id('001').set_color('#64a7bc') #pitch only
	v.get_patch_by_id('010').set_color('#78AB46') #spec only
	v.get_patch_by_id('101').set_color('#946774') #phnfeat + pitch intersection
	v.get_patch_by_id('110').set_color('#9e6a34') #phnfeat + spec intersection
	v.get_patch_by_id('011').set_color('#6ea981') #pitch + spec intersection
	v.get_patch_by_id('111').set_color('#8e7960') #all comb

	# #remove numbers from the venn diagram for each section 
	for idx, subset in enumerate(v.subset_labels):
		v.subset_labels[idx].set_visible(False)
	
	if save_fig:
			plt.savefig('%s/%s_vennDiagram_spec.pdf' %(save_dir, stimulus_class))
	return venn_diagram

def unique_variance_Barplot_spec(stimulus_class, subject_list, venn_diagram_spec, save_dir, opacity=0.4, save_fig=True):

	n_groups = len(subject_list)


	#plt.figure(figsize=(8,4))


	index = np.arange(n_groups)
	bar_width = 0.25


	unique_var_pitch = []
	unique_var_phnfeat = []
	unique_var_spec = []

	for idx, s in enumerate(subject_list):
		unique_var_pitch.append(venn_diagram[idx][2]) #spec_U_phnfeat
		unique_var_phnfeat.append(venn_diagram[idx][1]) #spec_U_pitch
		unique_var_spec.append(venn_diagram[idx][0]) #pitch_U_phnfeat

	df = pd.DataFrame(list(zip(unique_var_pitch, unique_var_phnfeat, unique_var_spec, np.array(unique_var_pitch) + np.array(unique_var_phnfeat) + np.array(unique_var_spec))), 
		columns=['pitch', 'phnfeat', 'spec', 'sum']) #create dataframe from list
	df.index = subject_list 

	my_colors = ['#64a7bc', '#cd1a1e', '#78AB46']
	df.sort_values(by='sum', ascending=False)[['pitch', 'phnfeat', 'spec']].plot(kind='bar', stacked=True, color=my_colors, alpha=opacity, width=0.5)


	plt.xlabel('Participant number')
	plt.ylabel('Unique variance')
	plt.title('Unique feature variance (%s)' %(stimulus_class))

	#plt.xticks(np.arange(len(subject_list)), np.array(subject_list), rotation=90)

	plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
	
	if save_fig:
		plt.savefig('%s/%s_uniqueVar_w_spec.pdf' %(save_dir, stimulus_class))
