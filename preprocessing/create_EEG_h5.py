from h5_tools import *

user = input('Enter computer username: ')

data_dir = '/Users/maansidesai/Box/MovieTrailersTask/Data/EEG/Participants' #path to participant audio/EEG data
textgrid_dir = '/Users/%s/Box/trailer_AV/textgrids/scene_cut_textGrids/' #path to textgrids
save_dir = '/Users/%s/Desktop/' #where to save large .h5 file

subject_list = ['MT0001', 'MT0002', 'MT0003', 'MT0004', 'MT0005', 'MT0006', 'MT0008',
				'MT0009', 'MT0010', 'MT0011', 'MT0012', 'MT0013', 'MT0014', 'MT0015', 'MT0016', 'MT0017']
print(subject_list)

wav_dirs = ['/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/Stimuli/TIMIT/',
			 '/Users/maansidesai/Desktop/UT_Austin/Research/Data_Analysis/Stimuli/MovieTrailers/' ] 

stimulus_class = ['TIMIT', 'MovieTrailers']

with h5py.File('%s/fullEEGmatrix.hf5'%(save_dir), 'w') as g:

	for idx, s in enumerate(stimulus_class):

		for subject in subject_list:
			raw = load_raw_EEG(subject, data_dir)
			new_fs = np.int(raw.info['sfreq'])
			evs, wav_id = load_event_file(s, subject, data_dir)
			unique_stimuli = np.unique(evs[:,2])
			unique_ids = np.unique(wav_id)


			if s == 'TIMIT':
				event_file = get_timit_phns_event_file(subject)
			else:
				event_file = get_trailer_phns_event_file(subject, 128.0)

			for wav_name, event_id in zip(unique_ids,unique_stimuli):
				epochs = get_event_epoch(raw, evs, event_id)
				
				#print(ep.shape)
				print(wav_name)
				print(event_id)
				binary_feat_mat, binary_phn_mat = binary_phn_mat_stim(subject, s, wav_name.split('.wav')[0], epochs)
				print(binary_feat_mat.shape)
				matrix=scene_cut(textgrid_dir, wav_name, ep, fs=128.0)
				g.create_dataset('%s/%s/resp/%s/epochs'%(s, wav_name, subject), data=np.array(epochs, dtype=float))
				try:
					g.create_dataset('%s/%s/stim/phn_feat_timings'%(s, wav_name), data=np.array(binary_feat_mat, dtype=float))
					g.create_dataset('%s/%s/stim/phn_timings'%(s, wav_name), data=np.array(binary_phn_mat, dtype=float))
					
				except:
					print('phn_timings already exists')
				
		# Now add stimulus representations
		   
				print('*********************************************')
				print(wav_name)
				print('*********************************************')
				
				if '%s/%s/stim/spec'%(s, wav_name) not in g:
				
					envelope = make_envelopes(wav_dirs[idx], wav_name, new_fs, epochs, pad_next_pow2=True)

					mel_spec, freqs = stimuli_mel_spec(wav_dirs[idx], wav_name)
					pitch_values = get_meanF0s_v2(fileName=wav_dirs[idx]+wav_name)
					binned_pitches = get_bin_edges_percent_range(pitch_values)


					#g.create_dataset('/%s/%s/stim/phn_time' %(s, wav_name), data=np.array(phoneme_times, dtype=float))  

					g.create_dataset('%s/%s/stim/spec'%(s, wav_name), data=np.array(mel_spec, dtype=float))
					g.create_dataset('%s/%s/stim/freqs'%(s, wav_name), data=np.array(freqs, dtype=float))
					g.create_dataset('%s/%s/stim/pitches'%(s, wav_name), data=np.array(pitch_values, dtype=float))
					g.create_dataset('%s/%s/stim/envelope'%(s, wav_name), data=np.array(envelope, dtype=float))
					g.create_dataset('%s/%s/stim/binned_pitches'%(s, wav_name), data=np.array(binned_pitches, dtype=float))
					g.create_dataset('%s/%s/stim/scene_cut'%(s, wav_name), data=np.array(matrix, dtype=float))


					print('all stims already exists')

