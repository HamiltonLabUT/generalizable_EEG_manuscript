import numpy as np
from scipy.signal import hanning, spectrogram, resample, hilbert, butter, filtfilt
from .fbtools import fft2melmx
from matplotlib import pyplot as plt

    
def get_envelope(audio, audio_fs, new_fs, cof=25):
    ''' Get the envelope of a sound file
    Inputs:
        w [float] : audio signal vector
        fs [int] : sampling rate of audio signal
        new_fs [int] : desired sampling rate of the envelope
    '''

    print("calculating hilbert transform")
    env_hilb = np.abs(hilbert(audio))

    nyq = audio_fs/2. #Nyquist frequency
    b, a = butter(3, cof/nyq, 'low'); #this designs a 3-pole low-pass filter
    
    print("Low-pass filtering hilbert transform to get audio envelope")
    envelope_long = np.atleast_2d(filtfilt(b, a, env_hilb, axis=0)) #filtfilt makes it non-causal (fwd/backward)

    envelope = resample(envelope_long.T, np.int(np.floor(envelope_long.shape[1]/(audio_fs/new_fs))))
    
    return envelope

def get_peak_rate(envelope):
    env_diff = np.diff(np.concatenate((0, envelope), axis=None))
    env_diff[env_diff<0] = 0

    return env_diff

def make_mel_spectrogram(w, fs, wintime=0.025, steptime=0.010, nfilts=80, minfreq=0, maxfreq=None):
    ''' Make mel-band spectrogram
    Inputs:
        w [float] : audio signal vector
        fs [int] : sampling rate of audio signal
        wintime [float] : window size
        steptime [float] : step size (time resolution)
        nfilts [int] : number of mel-band filters
        minfreq [int] : Minimum frequency to analyze (in Hz)
        maxfreq [int] : Maximum frequency to analyze (in Hz). If none, defaults to fs/2
    
    Outputs:
        mel_spectrogram [array]: mel-band spectrogram
        freqs [array] : array of floats, bin edges of spectrogram

    '''
    if maxfreq is None:
        maxfreq = np.int(fs/2)

    pspec, e = powspec(w, sr=fs, wintime=wintime, steptime=steptime, dither=1)
    aspectrum, wts, freqs = audspec(pspec, sr=fs, nfilts=nfilts, fbtype='mel', minfreq=minfreq, maxfreq=maxfreq, sumpower=True, bwidth=1.0)
    mel_spectrogram = aspectrum**0.001

    return mel_spectrogram, freqs

def powspec(x, sr=8000, wintime=0.025, steptime=0.010, dither=1):
    '''
    # compute the powerspectrum and frame energy of the input signal.
    # basically outputs a power spectrogram
    #
    # each column represents a power spectrum for a given frame
    # each row represents a frequency
    #
    # default values:
    # sr = 8000Hz
    # wintime = 25ms (200 samps)
    # steptime = 10ms (80 samps)
    # which means use 256 point fft
    # hamming window
    #
    # $Header: /Users/dpwe/matlab/rastamat/RCS/powspec.m,v 1.3 2012/09/03 14:02:01 dpwe Exp dpwe $

    # for sr = 8000
    #NFFT = 256;
    #NOVERLAP = 120;
    #SAMPRATE = 8000;
    #WINDOW = hamming(200);
    '''

    winpts = int(np.round(wintime*sr))
    steppts = int(np.round(steptime*sr))

    NFFT = 2**(np.ceil(np.log(winpts)/np.log(2)))
    WINDOW = hanning(winpts).T

    # hanning gives much less noisy sidelobes
    NOVERLAP = winpts - steppts
    SAMPRATE = sr

    # Values coming out of rasta treat samples as integers,
    # not range -1..1, hence scale up here to match (approx)
    f,t,Sxx = spectrogram(x*32768, nfft=NFFT, fs=SAMPRATE, nperseg=len(WINDOW), window= WINDOW, noverlap=NOVERLAP)
    y = np.abs(Sxx)**2

    # imagine we had random dither that had a variance of 1 sample
    # step and a white spectrum.  That's like (in expectation, anyway)
    # adding a constant value to every bin (to avoid digital zero)
    if dither:
        y = y + winpts

    # ignoring the hamming window, total power would be = #pts
    # I think this doesn't quite make sense, but it's what rasta/powspec.c does

    # 2012-09-03 Calculate log energy - after windowing, by parseval
    e = np.log(np.sum(y))

    return y, e 

def audspec(pspectrum, sr=16000, nfilts=80, fbtype='mel', minfreq=0, maxfreq=8000, sumpower=True, bwidth=1.0):
    '''
    perform critical band analysis (see PLP)
    takes power spectrogram as input
    '''

    [nfreqs,nframes] = pspectrum.shape

    nfft = int((nfreqs-1)*2)
    freqs = []

    if fbtype == 'mel':
        wts, freqs = fft2melmx(nfft=nfft, sr=sr, nfilts=nfilts, bwidth=bwidth, minfreq=minfreq, maxfreq=maxfreq);
    elif fbtype == 'htkmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 1);
    elif fbtype == 'fcmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 0);
    elif fbtype == 'bark':
        wts = fft2barkmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq);
    else:
        error(['fbtype ' + fbtype + ' not recognized']);

    wts = wts[:, 0:nfreqs]
    #figure(1)
    #plt.imshow(wts)

    # Integrate FFT bins into Mel bins, in abs or abs^2 domains:
    if sumpower:
        aspectrum = np.dot(wts, pspectrum)
    else:
        aspectrum = np.dot(wts, np.sqrt(pspectrum))**2.

    return aspectrum, wts, freqs
