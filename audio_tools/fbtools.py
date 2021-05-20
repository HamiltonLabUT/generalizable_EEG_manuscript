import numpy as np

#######################################
def mel2hz(z, htk=False):
#   f = mel2hz(z, htk)
#   Convert 'mel scale' frequencies into Hz
#   Optional htk = 1 means use the HTK formula
#   else use the formula from Slaney's mfcc.m
# 2005-04-19 dpwe@ee.columbia.edu


  if htk:
    f = 700.*(10**(z/2595.)-1);
  else:
    f_0 = 0.; # 133.33333;
    f_sp = 200./3.; # 66.66667;
    brkfrq = 1000.;
    brkpt  = (brkfrq - f_0)/f_sp;  # starting mel value for log region
    logstep = np.exp(np.log(6.4)/27.); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

    linpts = [z < brkpt]
    nonlinpts = [z >= brkpt]

    f = 0*z;

    # fill in parts separately
    f[linpts] = f_0 + f_sp*z[linpts];
    f[nonlinpts] = brkfrq*np.exp(np.log(logstep)*(z[nonlinpts]-brkpt));
  return f

####################################
def hz2mel(f,htk=False):
  '''
  #  z = hz2mel(f,htk)
  #  Convert frequencies f (in Hz) to mel 'scale'.
  #  Optional htk = 1 uses the mel axis defined in the HTKBook
  #  otherwise use Slaney's formula
  # 2005-04-19 dpwe@ee.columbia.edu
  '''

  if htk:
    z = 2595 * log10(1+f/700)
  else:
    # Mel fn to match Slaney's Auditory Toolbox mfcc.m

    f_0 = 0.; # 133.33333;
    f_sp = 200./3; # 66.66667;
    brkfrq = 1000.;
    brkpt  = (brkfrq - f_0)/f_sp;  # starting mel value for log region
    logstep = np.exp(np.log(6.4)/27.); # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

    linpts = [f < brkfrq];

    z = 0*f;

    # fill in parts separately
    if len(linpts) == 1:
      if linpts[0]:
        z = f - f_0/f_sp
      else:
        z = brkpt+(np.log(f/brkfrq))/np.log(logstep)
    else:
      z[linpts==True] = (f[linpts==True] - f_0)/f_sp
      z[linpts==False] = brkpt+(np.log(f[linpts==False]/brkfrq))/np.log(logstep)

  return z

def fft2melmx(nfft, sr=8000, nfilts=0, bwidth=1.0, minfreq=0, maxfreq=4000, htkmel=False, constamp=0):
  '''
  #      Generate a matrix of weights to combine FFT bins into Mel
  #      bins.  nfft defines the source FFT size at sampling rate sr.
  #      Optional nfilts specifies the number of output bands required 
  #      (else one per "mel/width"), and width is the constant width of each 
  #      band relative to standard Mel (default 1).
  #      While wts has nfft columns, the second half are all zero. 
  #      Hence, Mel spectrum is fft2melmx(nfft,sr)*abs(fft(xincols,nfft));
  #      minfreq is the frequency (in Hz) of the lowest band edge;
  #      default is 0, but 133.33 is a common standard (to skip LF).
  #      maxfreq is frequency in Hz of upper edge; default sr/2.
  #      You can exactly duplicate the mel matrix in Slaney's mfcc.m
  #      as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
  #      htkmel=1 means use HTK's version of the mel curve, not Slaney's.
  #      constamp=1 means make integration windows peak at 1, not sum to 1.
  #      frqs returns bin center frqs.
  # 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx
  '''
  
  if nfilts == 0:
    nfilts = np.ceil(hz2mel(maxfreq, htkmel)/2);
  
  wts = np.zeros((nfilts, nfft))
  
  # Center freqs of each FFT bin
  fftfrqs = np.arange(0,nfft/2.)/nfft*sr
  
  # 'Center freqs' of mel bands - uniformly spaced between limits
  minmel = hz2mel(minfreq, htkmel)
  maxmel = hz2mel(maxfreq, htkmel)
  binfrqs = mel2hz(minmel+np.arange(0.,nfilts+2)/(nfilts+2.)*(maxmel-minmel), htkmel);
  
  binbin = np.round(binfrqs/sr*(nfft-1.))
  
  for i in np.arange(nfilts):
    fs = binfrqs[i+[0, 1, 2]]

    #print fs
    # scale by width
    fs = fs[1]+bwidth*(fs - fs[1]);
    # lower and upper slopes for all bins
    loslope = (fftfrqs - fs[0])/(fs[1] - fs[0])
    hislope = (fs[2] - fftfrqs)/(fs[2] - fs[1])
    w = np.min((loslope, hislope), axis=0)
    w[w<0] = 0
    # .. then intersect them with each other and zero
    wts[i, 0:np.int(nfft/2)] = w
    
  if constamp == 0:
    # Slaney-style mel is scaled to be approx constant E per channel
    wts = np.dot(np.diag(2./(binfrqs[2+np.arange(nfilts)]-binfrqs[np.arange(nfilts)])),wts)
    #wts = np.atleast_2d(2/(binfrqs[2+np.arange(nfilts)]-binfrqs[np.arange(nfilts)]))*wts
    #wts = np.dot(2/(binfrqs[2+np.arange(nfilts)]-binfrqs[np.arange(nfilts)]),wts)
  
  # Make sure 2nd half of FFT is zero
  wts[:,np.int(nfft/2+2):np.int(nfft)] = 0
  # seems like a good idea to avoid aliasing
  return wts, binfrqs

