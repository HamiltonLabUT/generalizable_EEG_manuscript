#General features for fitting encoding models on EEG data

import scipy.io # For .mat files
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
import numpy as np
from numpy.polynomial.polynomial import polyfit
from audio_tools import spectools, fbtools, phn_tools
from scipy.io import wavfile
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
import glob
import re

from matplotlib import pyplot as plt
import parselmouth as pm
from parselmouth.praat import call
from ridge.utils import make_delayed, counter, save_table_file
from ridge.ridge import ridge_corr, bootstrap_ridge, bootstrap_ridge_shuffle, eigridge_corr

import random
import itertools as itools
np.random.seed(0)
random.seed(0)

from scipy import stats

#default argument for sampling rate as function input

def get_timit_phns_event_file(subj): 
	'''
	The individual subject textfiles are needed for TIMIT in order to generate the exact onset and offset times (in samples: 128Hz)

	'''
#subj = 'TD0002'
	phn_file = '/Users/maansidesai/Box/timit_decoding/data/event_files/TIMIT_phn_info_index.txt'
	subj_event_files = '/Users/maansidesai/Box/timit_decoding/data/%s/alignment_files_eeg/%s_all_timing_events.txt' %(subj, subj)

	#read into contents of the phoneme alignment file containing the .phn info in 128Hz sampling rate 
	sentence_idx = np.loadtxt(phn_file, dtype=np.int, usecols = (0))
	time_seconds = np.loadtxt(phn_file, dtype=np.float, usecols = (2))
	phoneme = np.loadtxt(phn_file, dtype=np.str, usecols = (3))
	phoneme_cat = np.loadtxt(phn_file, dtype=np.str, usecols = (4))
	sentence_name = np.loadtxt(phn_file, dtype=np.str, usecols = (5))
	phoneme_idx = np.loadtxt(phn_file, dtype=np.float, usecols = (6))

	#read into individual subject event file
	#first three var = now in 128Hz sampling rate 
	subj_onset = (np.loadtxt(subj_event_files, dtype=np.float, usecols = (1))/16000)*128
	subj_offset = (np.loadtxt(subj_event_files, dtype=np.int, usecols = (2))/16000)*128
	subj_noise_onset = (np.loadtxt(subj_event_files, dtype=np.float, usecols = (3))/16000)*128
	subj_sentence = np.loadtxt(subj_event_files, dtype=np.str, usecols = (4))
	subj_sent_idx = np.loadtxt(subj_event_files, dtype=np.int, usecols = (5))

	#separate out clean sentences vs. noisy sentences by index 
	clean_sentences = np.arange(0,20,2)
	noisy_sentences = np.arange(1,21,2)

	clean_onsets = []
	clean_offsets = []
	clean_timit_sentences = []
	clean_timit_idx = []

	noisy_onsets = []
	noisy_offsets = []
	noisy_timit_sentences = []
	noisy_timit_idx = []

	for ix, onset in enumerate(subj_sent_idx):
		if onset in clean_sentences:
			clean_onsets.append(subj_onset[ix])
			clean_offsets.append(subj_offset[ix])
			clean_timit_sentences.append(subj_sentence[ix])
			clean_timit_idx.append(subj_sent_idx[ix])
		elif onset in noisy_sentences:
			noisy_onsets.append(subj_noise_onset[ix])
			noisy_offsets.append(subj_offset[ix])
			noisy_timit_sentences.append(subj_sentence[ix])
			noisy_timit_idx.append(subj_sent_idx[ix])

	#stack separated clean vs. noisy data time points 
	clean_data = np.stack([clean_onsets, clean_offsets, clean_timit_sentences, clean_timit_idx], axis=1)
	noisy_data = np.stack([noisy_onsets, noisy_offsets, noisy_timit_sentences, noisy_timit_idx], axis=1)

	#initialized new phoneme time + category alignment for CLEAN TIMIT 
	phn_clean_timing_aln = [] #phoneme onset timing in 128Hz
	phn_clean_aln = [] #phoneme name itself 
	phn_cat_clean_aln = [] #category of phoneme 
	phn_clean_sent_name = []
	phn_clean_idx = []
	for idx, m in enumerate(clean_data):
		phn_clean_timing_aln.append(time_seconds[np.where(m[2] == sentence_name)] + clean_onsets[idx]) #add phon times to onset of clean TIMIT sentence
		phn_cat_clean_aln.append(phoneme_cat[np.where(m[2] == sentence_name)]) #add phon cat to each phoneme 
		phn_clean_aln.append(phoneme[np.where(m[2] == sentence_name)])
		phn_clean_sent_name.append(sentence_name[np.where(m[2] == sentence_name)])
		phn_clean_idx.append(phoneme_idx[np.where(m[2] == sentence_name)])

	#initialized new phoneme time + category alignment for NOISY TIMIT 
	phn_noisy_timing_aln = [] #phoneme onset timing in 128Hz
	phn_noisy_aln = [] #phoneme itself 
	phn_cat_noisy_aln = [] #category of phoneme 
	phn_noisy_sent_name = [] #???? FIX THIS!
	phn_noisy_idx = []
	for idx, n in enumerate(noisy_data):
		#noisy_timit = []
		phn_noisy_timing_aln.append(time_seconds[np.where(n[2][:-9] == sentence_name)] + noisy_onsets[idx]) #add phon times to onset of clean TIMIT sentence
		phn_cat_noisy_aln.append(phoneme_cat[np.where(n[2][:-9] == sentence_name)]) #add phon cat to each phoneme 
		phn_noisy_aln.append(phoneme[np.where(n[2][:-9] == sentence_name)])
		phn_noisy_sent_name.append(sentence_name[np.where(n[2][:-9] == sentence_name)]) 
		phn_noisy_idx.append(phoneme_idx[np.where(n[2][:-9] == sentence_name)])

		# for sent in noisy_timit_sentences:
		# 	#noisy_timit.append(n[2])
		# 	phn_noisy_sent_name.append(sent)

	#convert to numpy array + stack:
	phn_clean_timing_aln = np.concatenate(phn_clean_timing_aln)
	phn_clean_aln = np.concatenate(phn_clean_aln)
	phn_cat_clean_aln = np.concatenate(phn_cat_clean_aln)
	phn_clean_sent_name = np.concatenate(phn_clean_sent_name)
	phn_clean_idx = np.concatenate(phn_clean_idx)

	phn_noisy_timing_aln = np.concatenate(phn_noisy_timing_aln)
	phn_noisy_aln = np.concatenate(phn_noisy_aln)
	phn_cat_noisy_aln = np.concatenate(phn_cat_noisy_aln)
	phn_noisy_idx = np.concatenate(phn_noisy_idx)

	phn_noisy_sent_name_all = [] #already concatenated... 
	for i, t in enumerate(phn_noisy_sent_name):
		string = np.array(('_6w_noise'))
		phn_noisy_sent_name_all.append(np.char.add(phn_noisy_sent_name[i], string))

	phn_noisy_sent_name_all=np.concatenate(phn_noisy_sent_name_all)
	#add in new phoneme and cat alignment for clean + noisy stim, overwrite previous variables:
	clean_data = np.stack([phn_clean_aln, phn_clean_timing_aln, phn_cat_clean_aln, phn_clean_sent_name, phn_clean_idx], axis=1)
	noisy_data = np.stack([phn_noisy_aln, phn_noisy_timing_aln, phn_cat_noisy_aln, phn_noisy_sent_name_all, phn_noisy_idx], axis=1)

	np.savetxt('/Users/maansidesai/Box/timit_decoding/data/event_files/%s_cleanTIMIT.txt'%(subj), clean_data, fmt='%s\t', delimiter='\t')
	np.savetxt('/Users/maansidesai/Box/timit_decoding/data/event_files/%s_noisyTIMIT.txt'%(subj), noisy_data, fmt='%s\t', delimiter='\t')

	return clean_data, noisy_data


#load EEG ICA data 
def load_raw_EEG(subj, data_dir):
	'''
	Load EEG preprocessing (ICA) data using MNE
	Identifies bad timepoints in the data from the annotations marked 

	Parameters
	----------
	subj : string 
	data_dir : string 

	Returns
	------
	raw : mne-format
		- MNE-python format of data loaded for specified subject
	'''
	eeg_file = '%s/%s/downsampled_128/%s_postICA_rejected.fif'%(data_dir, subj, subj)
	raw = mne.io.read_raw_fif(eeg_file, preload=True)

	# Print which are the bad channels, but don't get rid of them yet...
	raw.pick_types(eeg=True, meg=False, exclude=[])
	bad_chans = raw.info['bads']
	print("Bad channels are: ")
	print(bad_chans)

	# Get onset and duration of the bad segments in samples
	bad_time_onsets = raw.annotations.onset * raw.info['sfreq']
	bad_time_durs = raw.annotations.duration * raw.info['sfreq']

	print(raw._data.shape)

	# Set the bad time points to zero
	for bad_idx, bad_time in enumerate(bad_time_onsets):
		raw._data[:,np.int(bad_time):np.int(bad_time+bad_time_durs[bad_idx])] = 0
	
	return raw

#function to epoch your data
def get_event_epoch(raw, evs, event_id, bef_aft=[0,0], baseline = None, reject_by_annotation=False):
	'''
	Epoch the data based on the data loaded for the participant and the event time points from the alignment function from the textfiles above

	'''

	# Get duration information
	max_samp_dur = np.max(evs[(np.where(evs[:,2] == event_id)),1]-evs[(np.where(evs[:,2] == event_id)),0])
	trial_dur = max_samp_dur/raw.info['sfreq']
	
	epochs = mne.Epochs(raw, evs, event_id=[event_id], tmin=bef_aft[0], tmax=trial_dur+bef_aft[1], baseline=baseline,
						reject_by_annotation=reject_by_annotation)
	ep = epochs.get_data()
		
	return ep

#load event file to get times of TIMIT sentences (clean and noisy)
def load_event_file(condition, subj, data_dir='/Users/maansidesai/Box/timit_decoding/data/'):
	event_file = '%s/%s/alignment_files_eeg/%s_all_timing_events.txt'%(data_dir, subj, subj)
	index = np.loadtxt(event_file, dtype='int', usecols = (5))
	wav_id = np.loadtxt(event_file, dtype='<U', usecols = 4)
	if condition=='clean':
		clean_sentences = np.arange(0,20,2)
		evs = np.loadtxt(event_file, dtype='f', usecols = (1, 2, 5))
		evs[:,:2] = (evs[:,:2]/16000)*128 #128 is the downsampled frequency from EEG data
		evs = evs.astype(np.int)
		clean_timit_sent = [] #take only clean TIMIT info 
		clean_evs = []
		for ix, val in enumerate(index): 
			if val in clean_sentences: 
				clean_timit_sent.append(wav_id[ix])
				clean_evs.append(evs[ix])
		wav_id = np.asarray(clean_timit_sent)
		evs = np.asarray(clean_evs)
	else:
		noisy_sentences = np.arange(1,21,2)
		evs = np.loadtxt(event_file, dtype='f', usecols = (3, 2, 5)) #this is to read in TIMIT from noisy condition because onset is different
		evs[:,:2] = (evs[:,:2]/16000)*128 
		evs = evs.astype(np.int) 
		noisy_timit_sent = [] #need to take only noisy condition info 
		noisy_evs = []
		for ix, val in enumerate(index): 
			if val in noisy_sentences: 
				noisy_timit_sent.append(wav_id[ix])
				noisy_evs.append(evs[ix])
		wav_id = np.asarray(noisy_timit_sent)
		evs = np.asarray(noisy_evs)
	
	return evs, wav_id


#create binary phoneme matrix
def binary_phn_mat_stim(subj, stimulus_class, wav_name, ep, evs, wav_id, file_path='/Users/maansidesai/Box/timit_decoding/data/event_files/'):

	evs_orig = evs.copy()
	evnames = wav_id.copy()
	#read into phoneme file 
	clean_file = '%s/TIMIT_phn_info_index.txt' %(file_path) # need to read in phoneme information and timings from textfile or whatever format you have (transcriptions with timing from neural data)
	time_samples = np.loadtxt(clean_file, dtype=np.float, usecols = (2)) #onset time
	time_samples = time_samples.astype(np.int)
	phoneme = np.loadtxt(clean_file, dtype=np.str, usecols = (3)) #phoneme
	phoneme_cat = np.loadtxt(clean_file, dtype=np.str, usecols = (4)) #phoneme category
	sentence_name = np.loadtxt(clean_file, dtype=np.str, usecols = (5)) #name of sentence

	#convert from samples to seconds:
	phn_seconds = time_samples

	phn1 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 
	'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 
	'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 
	't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
	
	assign_num = {i: idx for idx, i in enumerate(phn1)}
	idx_num = [assign_num[i] for i in phn1]

	timing = dict()
	binary_phn_mat = dict()

	timing[wav_name] = []
	mat_length = ep.shape[2]
	print(mat_length)
	binary_phn_mat = np.zeros((len(np.unique(phn1)), mat_length))
	print(binary_phn_mat.shape)

	for i, s in enumerate(sentence_name):
		if s == wav_name:
			phn_time = time_samples[i]
			phn_time = int(phn_time)
			timing[wav_name].append(phn_time)
			timit_phn = phoneme[i]	
			if timit_phn in phn1:
				phoneme_idx = assign_num[timit_phn]	
				binary_phn_mat[phoneme_idx, phn_time] = 1

	binary_feat_mat, fkeys = phn_tools.convert_phn(binary_phn_mat.T, 'features')

	return binary_feat_mat.T, binary_phn_mat

#create envelopes
def make_envelopes(wav_dir, wav_name, new_fs, ep, pad_next_pow2=True):  
	'''
	Create acoustic envelope 
	'''

	print("Sentence: %s"% (wav_name))
	wfs, sound = wavfile.read('%s/%s.wav'%(wav_dir, wav_name))
	sound = sound/sound.max()
	envelopes = []
	envelope = spectools.get_envelope(sound, wfs, new_fs, pad_next_pow2=pad_next_pow2)

	return envelope

def stimuli_mel_spec(path, wav_name):
	'''
	Create mel spectrogram 
	'''
	[fs,w] = wavfile.read(path+'/'+ wav_name)
	w=w.astype(np.float)
	
	mel_spec, freqs = spectools.make_mel_spectrogram(w, fs, wintime=0.025, steptime=1/128.0, nfilts=80, minfreq=0, maxfreq=None)

	return mel_spec, freqs

def get_meanF0s_v2(fileName, steps=1/128.0, f0min=50, f0max=300):
	"""
	Uses parselmouth Sound and Pitch object to generate frequency spectrum of
	wavfile, 'fileName'.  Mean F0 frequencies are calculated for each phoneme
	in 'phoneme_times' by averaging non-zero frequencies within a given
	phoneme's time segment.  A range of 10 log spaced center frequencies is
	calculated for pitch classes. A pitch belongs to the class of the closest
	center frequency bin that falls below one standard deviation of the center
	frequency range.
	​
	"""
	#fileName = wav_dirs + wav_name
	sound =  pm.Sound(fileName)
	pitch = sound.to_pitch(steps, f0min, f0max) #create a praat pitch object
	pitch_values = pitch.selected_array['frequency']
	
	return pitch_values


#binned pitch
def get_bin_edges_percent_range(a, bins=10, percent=95):
    ''' inputs:
        a [vector over time] : your pitch values in Hz
        bins [int]: number of bins you want to separate those pitches into
        percent [int]: percentile so you ignore extreme values
        ## PROBABLY DONT USE THIS BUT YOU COULD, JUST NEED OVER ALL PITCHES
    '''
    assert percent > 1 
    assert percent < 100
    tail_percentage = (100 - percent)/2
    a_range = np.percentile(a, [tail_percentage, 100-tail_percentage])
    counts, bin_edges = np.histogram(a, bins=bins, range=a_range)
    return bin_edges

# binned_edges = np.linspace(50,300,15)
def get_pitch_matrix(pitch, bin_edges=np.linspace(50,300,15)):
    pitch[pitch < bin_edges[0]] = bin_edges[0] + 0.0001
    pitch[pitch > bin_edges[-1]] = bin_edges[-1] - 0.0001
    bin_indexes = np.digitize(pitch, bin_edges) - 1
    stim_pitch = np.zeros((len(pitch), 10))
    for i, b in enumerate(bin_indexes):
        if b < 10:
            stim_pitch[i, b] = 1
    return stim_pitch
