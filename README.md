# generalizable_EEG_manuscript
Code for Desai et al. 2021 (Generalizable EEG encoding models)
Data: https://osf.io/p7qy8/?view_only=bd1ca019ba08411fac723d48097c231d

Jupyter Notebooks generates the figures in paper and the panels. 

Each notebook reads from `run_STRF_analysis.py` which is a python file with all of the functions 

- List of notebooks below and what they all do:

1) `encodingModel_generation.ipynb` : to fit individual or a combination of acoustic and linguistic features for all participants (n=16) and for both of the constrasting stimuli (TIMIT and Movie Trailers)
2) `Figure2_acoustic_linguisticRepresentations.ipynb` : Figure 2 plots of manuscript
3) `Figure3_CrossPredictionAnalysis.ipynb` : Figure 3 plots of manuscript (cross-correlation analysis)
4) `Figure4_ModelPerformance_SlidingScaleCorrelation.ipynb` : Figure 4 of manuscript (amount of testing data for TIMIT and trailers as well as segmented analysis.
5) `Figure5_Gabor_Stimuli_Examples.ipynb` : visual encoding models (alone and in combination with auditory full model) 


audio tools : folder which contains relevant functions to create acoustic features 
preprocessing : folder which creates large .h5 file, called fullEEGmatrix.h5 which contains all of the extracted stimulus features (acoustic, which includes the envelope, binned pitch, and spectrogram and linguistic, phonlogical features, and visual features [scene cuts]). The gabor wavelets were created in the notebook Figure5_Gabor_Stimuli_Examples.ipynb. 

ridge : folder which contains the toolbox for fitting the regression/encoding model 
