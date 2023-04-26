# generalizable_EEG_manuscript
Code for Desai et al. 2021 [Generalizable EEG Encoding Models with Naturalistic Audiovisual Stimuli](https://www.jneurosci.org/content/41/43/8946?utm_source=TrendMD&utm_medium=cpc&utm_campaign=JNeurosci_TrendMD_1). Journal of Neuroscience 27 October 2021, 41 (43) 8946-8962; DOI: https://doi.org/10.1523/JNEUROSCI.2891-20.2021

## Abstract
In natural conversations, listeners must attend to what others are saying while ignoring extraneous background sounds. Recent studies have used encoding models to predict electroencephalography (EEG) responses to speech in noise-free listening situations, sometimes referred to as “speech tracking.” Researchers have analyzed how speech tracking changes with different types of background noise. It is unclear, however, whether neural responses from acoustically rich, naturalistic environments with and without background noise can be generalized to more controlled stimuli. If encoding models for acoustically rich, naturalistic stimuli are generalizable to other tasks, this could aid in data collection from populations of individuals who may not tolerate listening to more controlled and less engaging stimuli for long periods of time. We recorded noninvasive scalp EEG while 17 human participants (8 male/9 female) listened to speech without noise and audiovisual speech stimuli containing overlapping speakers and background sounds. We fit multivariate temporal receptive field encoding models to predict EEG responses to pitch, the acoustic envelope, phonological features, and visual cues in both stimulus conditions. Our results suggested that neural responses to naturalistic stimuli were generalizable to more controlled datasets. EEG responses to speech in isolation were predicted accurately using phonological features alone, while responses to speech in a rich acoustic background were more accurate when including both phonological and acoustic features. Our findings suggest that naturalistic audiovisual stimuli can be used to measure receptive fields that are comparable and generalizable to more controlled audio-only stimuli.

## Data
Data can be found at [OSF](https://osf.io/p7qy8/?view_only=bd1ca019ba08411fac723d48097c231d).

## How to Run
Jupyter Notebooks can be used to generate the figures in the paper.

Each notebook reads from `run_STRF_analysis.py` which is a python file with all of the functions that analyze the data.

### List of notebooks below and what they all do:

1) `encodingModel_generation.ipynb` : to fit individual or a combination of acoustic and linguistic features for all participants (n=16) and for both of the constrasting stimuli (TIMIT and Movie Trailers)
2) `Figure2_acoustic_linguisticRepresentations.ipynb` : Figure 2 plots of manuscript
3) `Figure3_CrossPredictionAnalysis.ipynb` : Figure 3 plots of manuscript (cross-correlation analysis)
4) `Figure4_ModelPerformance_SlidingScaleCorrelation.ipynb` : Figure 4 of manuscript (amount of testing data for TIMIT and trailers as well as segmented analysis.
5) `Figure5_Gabor_Stimuli_Examples.ipynb` : visual encoding models (alone and in combination with auditory full model) 

### Folders:

`audio tools` : folder which contains relevant functions to create acoustic features 
preprocessing : folder which creates large .h5 file, called fullEEGmatrix.h5 which contains all of the extracted stimulus features (acoustic, which includes the envelope, binned pitch, and spectrogram and linguistic, phonlogical features, and visual features [scene cuts]). The gabor wavelets were created in the notebook Figure5_Gabor_Stimuli_Examples.ipynb. 

`ridge` : folder which contains the toolbox for fitting the regression/encoding model 
