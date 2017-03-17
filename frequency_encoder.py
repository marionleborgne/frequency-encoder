# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np
import warnings

from nupic.data import SENTINEL_VALUE_FOR_MISSING_DATA
from nupic.encoders.base import Encoder
from nupic.encoders.scalar import ScalarEncoder



class FrequencyEncoder(Encoder):
  def __init__(self, numFrequencyBins, freqBinN, freqBinW, minval=0,
               maxval=30.0, log=True, normalize=False, clipWithWarning=True):
    """
    The `FrequencyEncoder` encodes a time series chunk (or any 1D array of
    numeric values) by taking the power spectrum of the signal and
    discretizing it. The discretization is done by slicing the frequency axis
    of the power spectrum in frequency bins. The parameter controlling the
    number of frequency bins is `numFrequencyBins`. The maximum amplitude of
    the power spectrum in this `frequencyBin` is encoded by a `ScalarEncoder`.
    The parameter in `FrequencyEncoder` controlling the frequency bin size
    is `freqBinN`, which corresponds to the parameter `n` of `ScalarEncoder`.
    The parameter in `FrequencyEncoder` controlling the resolution (width)
    of a bin size is `freqBinW`, which corresponds to the parameter
    `w` of `ScalarEncoder`.


    :param numFrequencyBins: (int) The number of each frequency bin used to
      discretize the power spectrum.

    :param freqBinN: (int) The size of each frequency bin in
      the power spectrum. This determines the 'n' parameter of the
      ScalarEncoder used to encode each frequency bin.

    :param freqBinW: (int) The resolution of each frequency bin in
      the power spectrum. This determines the 'w' parameter of the
      ScalarEncoder used to encode each frequency bin.

    :param minval: (float) optional. The minimum value of the power spectrum.
      This determines the 'minval' parameter of the ScalarEncoder used to
      encode each frequency bin. In practice, the power spectrum is always
      positive, so minval=0 works well.

    :param maxval: (float) optional. The maximum value allowed in the power
      spectra. This determines the maxval parameter of the ScalarEncoder used to
      encode each frequency bin. The default value of maxval=30.0 is based on
      16-bit samples from CD recordings, with log-transformation and no
      normalization (see below).  For different encodings, different values will
      be more appropriate.  Ideally, this value should be as low as possible to
      maximize resolution.

    :param log: (bool) optional. If True (default), the power spectrum is
      log-transformed before encoding. This flattens the distribution of values
      and allows for a generic default value of maxval that should accomodate
      most encodings. It also "compresses" the higher range of power values, which 
      empirically improves classification in some cases. 
      Note that it may be a good idea to scale your signal to take advantage of the 
      log-transform: inputs with larger values will produce larger power spectrum values,
      which will undergo more compression in the high range, while smaller signals will 
      remain closer to the linear-like portion of the logarithmic curve. Stronger 
      compression may improve classification performance in some cases (e.g. try to 
      see what happens if you scale your signal within the [-1000, 1000] range).
      If you set this parameter to False, the distribution of power spectrum values
      will become very long-tailed and you will need to tune the maxval parameter. 

    :param normalize: (bool) optional. Normalizes the power spectrum to fit the entire 
      [minval, maxval] range. This may maximize resolution and avoids exceeding maxval, 
      but loses information about the actual energy in the encoded sample, making all
      encoded samples similarly "loud". 
      Note that this is applied after any log-transforming.

    :param clipWithWarning: (bool) optional. If True, power spectrum values above maxval
      will be clipped to maxval before being fed to ScalarEncoder and a warning will be 
      raised. If False, the power spectra are fed "as is" to ScalarEncoder, and values
      above maxval cause an error.
    """

    self.numFrequencyBins = numFrequencyBins
    self.freqBinN = freqBinN
    self.freqBinW = freqBinW
    self.minval = minval
    self.maxval = maxval
    self.log = log
    self.normalize = normalize
    self.clipWithWarning= clipWithWarning

    self.outputWidth = numFrequencyBins * freqBinN
    self.scalarEncoder = ScalarEncoder(n=freqBinN,
                                       w=freqBinW,
                                       minval=minval,
                                       maxval=maxval,
                                       forced=True)


  def getWidth(self):
    """
    Return the output width, in bits.

    :return outputWidth:  (int) output width
    """
    return self.outputWidth


  def encodeIntoArray(self, inputData, output):
    """
    Encodes inputData and puts the encoded value into the numpy output array,
    which is a 1D array of length returned by getWidth().

    :param inputData: (np.array) Data to encode.
    :param output: (np.array) 1D array. Encoder output.
    """

    if type(inputData) != np.ndarray:
      raise TypeError('Expected inputData to be a numpy array but the input '
                      'type is %s' % type(inputData))

    if inputData is SENTINEL_VALUE_FOR_MISSING_DATA:
      output[0:self.outputWidth] = 0
    else:
      freqs = getFreqs(inputData, self.log, self.normalize, self.minval, self.maxval)
      if self.clipWithWarning:
          nbabovemax = np.sum(freqs > self.maxval)
          if nbabovemax > 0:
              warnings.warn(str(nbabovemax)+" out of "+str(len(freqs))+" power spectrum values higher than maximum allowed value "+str(self.maxval)+", clipping...")
              freqs[freqs > self.maxval] = self.maxval

      freqBinSize = len(freqs) / self.numFrequencyBins
      binEncodings = []
      for i in range(self.numFrequencyBins):
        freqBin = freqs[i * freqBinSize:(i + 1) * freqBinSize]
        binVal = np.max(freqBin)
        binEncoding = self.scalarEncoder.encode(binVal)
        binEncodings.append(binEncoding.tolist())

      output[0:self.outputWidth] = np.array(binEncodings).flatten()



def getFreqs(data, log=True, normalize=False, minval=0.0, maxval=30.0):
  """
  Get the FFT of a 1-D array.

  :param data: (np.array) input signal to analyze.
  :param log: (bool) whether to take the log of the power spectrum. Taking
      the log minimizes variations in maximum power, normalizes overall 
      distribution of powers and squashes resolution at higher energies, 
      which may actually help classification.
  :param normalize: (bool) whether to normalize the power spectrum within 
      the [minval, maxval] range before encoding (but after log-transform).
  :param minval: (scalar) minimum value to use in the normalization. 
  :param maxval: (scalar) maximum value to use in the normalization. 
  :return fft: (np.array) the power spectrum of the input signal.
  """
  fft = (abs(np.fft.rfft(data)) ** 2) / len(data)

  if log:
    fft =  np.log(1 + fft)
  
  if normalize:
      fft = fft - np.min(fft)
      fft = fft / np.max(fft)  # now fft is normalized to [0,1]
      fft = minval + fft * (maxval-minval) # now fft is normalized to [minval, maxval]

  return fft



def pprint(encoding, numFrequencyBins, freqBinSize):
  """
  Output a pretty-printed version of a frequency encoding.

  :param encoding: (np.array) Output of a FrequencyEncoder
  :param numFrequencyBins: (int) number of frequency bins of the Encoder.
  :param freqBinSize: (int) size of each frequency bin.
  """
  for i in range(numFrequencyBins):
    print ('Freq bin %s: '
           '%s' % (i, encoding[i * freqBinSize:(i + 1) * freqBinSize]))
