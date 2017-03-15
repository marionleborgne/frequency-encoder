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

from nupic.data import SENTINEL_VALUE_FOR_MISSING_DATA
from nupic.encoders.base import Encoder
from nupic.encoders.scalar import ScalarEncoder



class FrequencyEncoder(Encoder):
  def __init__(self, numFrequencyBins, freqBinN, freqBinW, minval=0,
               maxval=14.0, log=True):
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

    :param maxval: (float) optional. The maximum value of the power spectrum.
      This determines the maxval parameter of the ScalarEncoder used to
      encode each frequency bin. After analysis, we found that by taking the
      log of the power spectrum allows us to use a default value of maval=14.0.

    :param log: (bool) whether to take the log of the power spectrum.
      Note: It is not recommended to set this to False. Taking the log dampens
      the amplitude variations of the power spectrum and allows us (after
      analysis) to set maxval to the default value of 14.0. If you use
      log=False, you will have to tune the maxval value.
    """

    self.numFrequencyBins = numFrequencyBins
    self.freqBinN = freqBinN
    self.freqBinW = freqBinW
    self.minval = minval
    self.maxval = maxval
    self.log = log

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
      freqs = getFreqs(inputData, self.log)
      freqBinSize = len(freqs) / self.numFrequencyBins
      binEncodings = []
      for i in range(self.numFrequencyBins):
        freqBin = freqs[i * freqBinSize:(i + 1) * freqBinSize]
        binVal = np.max(freqBin)
        binEncoding = self.scalarEncoder.encode(binVal)
        binEncodings.append(binEncoding.tolist())

      output[0:self.outputWidth] = np.array(binEncodings).flatten()



def getFreqs(data, log=True):
  """
  Get the FFT of a 1-D array.

  :param data: (np.array) input signal to analyze.
  :param log: (bool) whether to take the log of the power spectrum. Taking
      the log with minimize amplitude variations.
  :return fft: (np.array) the power spectrum of the input signal.
  """
  fft = abs(np.fft.rfft(data)) ** 2
  if log:
    return np.log(1 + fft)
  else:
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
