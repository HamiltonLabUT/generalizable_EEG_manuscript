from run_STRF_analysis import *
mpl.use('Agg')


#data_dir='/Users/%s/Box/MovieTrailersTask/Data/EEG/Participants' %(user)
data_dir = os.path.abspath('../data')

subject_list = create_subject_list(data_dir+'/participants')

stimulus_class = ['MovieTrailers', 'TIMIT']


for i in stimulus_class:
	for idx, ii in enumerate(subject_list):
		wt, corrs, valphas, allRcorrs, all_corrs_shuff = strf_features(ii, i, data_dir, save_dir, fs=128.0, delay_max=0.6, delay_min=0.0, wt_pad=0.0, 
			full_gabor = False, full_audio_spec = False, full_model = False, pitchUenvs = False, pitchUphnfeat = False, envsUphnfeat = False, 
			phnfeat_only = False, envs_only = True, pitch_only = False, gabor_only = False, spec = False, pitchphnfeatspec = False, 
			binned_pitch_full_audio = True, binned_pitch_envs=False, binned_pitch_phnfeat=False, binned_pitch_full_audiovisual=False, 
			binned_pitch_only=False, pitchUspec=False, phnfeatUspec=False, spec_scaled=False, pitchphnfeatspec_scaled=False, scene_cut=False, 
			scene_cut_gaborpc=False)