# generalizable_EEG_manuscript
Code for Desai et al. 2021 (Generalizable EEG encoding models)

Jupyter Notebooks generates the figures in paper and the panels. 

Each notebook reads from `run_STRF_analysis.py` which is a python file with all of the functions 

The data for all of the individual subject encoding models can be found in (insert link for data here). However to generate all of the subject files (.h5 individual files for each auditory or visual or combined model type, use `STRF_allSubj.py` which also reads from `run_STRF_analysis.py. However, toggle the appropriate flag in `STRF_allSubj.py` to run the specific encoding model for each subject to generate the desired model)

audio tools : folder which contains relevant functions to create acoustic features 
preprocessing : folder which creates large .h5 file, called fullEEGmatrix.h5 which contains all of the extracted stimulus features (acoustic, which includes the envelope, binned pitch, and spectrogram and linguistic, phonlogical features, and visual features [scene cuts]). The gabor wavelets were created in the notebook Figure5_Gabor_Stimuli_Examples.ipynb. 

ridge : folder which contains the toolbox for fitting the regression/encoding model 