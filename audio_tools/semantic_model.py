from gensim.models import word2vec
import gensim
import h5py
import numpy as np

print("Loading word2vec model from Google News trained dataset")
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/liberty/Documents/UCSF/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

subjects = ['EC2','EC6','EC33','EC35','EC37','EC52','EC55','EC56','EC60','EC61','EC63']
subjects = ['EC82','EC84','EC92','EC118','EC129']
subjects = ['EC53','EC75','EC85','EC89','EC124','EC143','EC157','EC180','EC193']
#subjects = ['EC53']
for subj in subjects:
	print("Making STRFs for %s"%(subj))
	if subj is 'EC53' or subj is 'EC85' or subj is 'EC89' or subj is 'EC143':
		nchans = 288
	elif subj is 'EC75' or subj is 'EC124':
		nchans = 64
	elif subj is 'EC157':
		nchans = 302
	elif subj is 'EC193':
		nchans = 274
	elif subj is 'EC180':
		nchans = 160
	else:
		nchans = 256
	train_file = '/Users/liberty/Documents/UCSF/data/timit/%s/strfs/ecog%d_log/TrainStimResp.hf5'%(subj, nchans)
	val_file = '/Users/liberty/Documents/UCSF/data/timit/%s/strfs/ecog%d_log/ValStimResp.hf5'%(subj, nchans)

	sound_dir = '/Users/liberty/Documents/UCSF/matlab/TIMIT/@ECSpeech/Sounds'

	ft=h5py.File(train_file)
	fv=h5py.File(val_file)

	sound_fs = ft['soundfs'][:][0]
	data_fs = ft['fs'][:][0]

	f = [train_file, val_file]

	for ff in f:
		fh = h5py.File(ff, 'r+')
		print("Finding semantic vectors for sentences in %s"%ff)

		all_semantic = []
		all_semantic_diff = []
		for s in fh['names']:
			
			# Figure out how long this stimulus is
			for line in open('%s/%s.txt'%(sound_dir, s)):
				dat = line.split(' ')
			start_samp = np.int(dat[0])
			stop_samp = np.int(dat[1])
			
			# Initialize semantic feature matrix for 300D word2vec vectors over time
			semantic_feats = np.zeros((np.int(np.round(stop_samp/sound_fs*data_fs))-1, 300 ))
			semantic_diff = np.zeros((np.int(np.round(stop_samp/sound_fs*data_fs))-1, 1 ))

			last_semantic_vec = np.zeros((300,))
			# Loop through all words, find word2vec representation and add to matrix
			for line in open('%s/%s.wrd'%(sound_dir, s)):
				[start, stop, wrd] = line.split(' ')
				wrd = wrd.rstrip() # Get rid of newline character if it's there
				if wrd[-2:] == "'s": # Get rid of possessives
					wrd = wrd[:-2]
				if wrd=="i'm":
					wrd = "im"
				if wrd=="i've":
					wrd = "ive"
				if wrd=="mudwagon":
					wrd = "wagon"
				if wrd[-1:] == "'":
					wrd = wrd[:-1]
				start_samp = np.int(np.round(np.int(start)/sound_fs*data_fs))
				try:
					semantic_feats[start_samp, : ] = model[wrd]
					semantic_diff[start_samp] = 1 - np.corrcoef(last_semantic_vec, model[wrd])[0,1]
					last_semantic_vec += model[wrd]
				except:
					print ("Word %s not in dict"%(wrd))

			all_semantic.append(semantic_feats)
			all_semantic_diff.append(semantic_diff)

		semantic_onsets = np.vstack((all_semantic))
		semantic_diffs = np.vstack((all_semantic_diff))
		semantic_diffs[np.where(np.isnan(semantic_diffs))]
		print(semantic_onsets.shape)
		print(semantic_diffs.shape)

		print("Saving data in %s"%ff)
		#fh.create_dataset('semantic_onset', semantic_onsets.shape, dtype='f8', data=semantic_onsets)
		fh.create_dataset('semantic_dissimilarity', semantic_diffs.shape, dtype='f8', data=semantic_diffs)
		fh.close()