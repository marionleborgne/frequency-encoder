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
from matplotlib import mlab

from nupic.data import SENTINEL_VALUE_FOR_MISSING_DATA

from nupic.encoders.base import Encoder
from nupic.encoders.scalar import ScalarEncoder


class FrequencyEncoder(Encoder):
    def __init__(self, frequencyCutoff, numFrequencyBins, freqBinN,
                 freqBinW, minval, maxval, normalize):

        self.frequencyCutoff = frequencyCutoff
        self.numFrequencyBins = numFrequencyBins
        self.freqBinN = freqBinN
        self.freqBinW = freqBinW
        self.minval = minval
        self.maxval = maxval
        self.normalize = normalize


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
        which is a 1-D array of length returned by getWidth().

        Note: The numpy output array is reused, so clear it before updating it.

        @param inputData Data to encode. This should be validated by the encoder.
        @param output numpy 1-D array of same length returned by getWidth()
        """

        if type(inputData) != np.ndarray:
            raise TypeError(
                "Expected a list or numpy array but got input of type %s" % type(
                    inputData))


        if inputData is SENTINEL_VALUE_FOR_MISSING_DATA:
          output[0:self.outputWidth] = 0
        else:
            freqs = abs(np.fft.rfft(inputData))**2
            if self.normalize and np.max(freqs) > 0: freqs /= np.max(freqs)
            freqBinSize = len(freqs)/self.numFrequencyBins

            binEncodings = []
            for i in range(self.numFrequencyBins):
                freqBin = freqs[i * freqBinSize:(i+1)*freqBinSize]
                meanFreq = np.max(freqBin)
                binEncoding = self.scalarEncoder.encode(meanFreq)
                binEncodings.append(binEncoding.tolist())

            output[0:self.outputWidth] = np.array(binEncodings).flatten()

def pprint(encoding, numFrequencyBins, freqBinSize):
    for i in range(numFrequencyBins):
        print ('Freq bin %s: '
               '%s' %(i,encoding[i*freqBinSize:(i+1)*freqBinSize]))


if __name__ == '__main__':


    frequencyCutoff = 30
    numFrequencyBins = 5
    freqBinN = 5
    freqBinW = 1
    minval = 0.0
    maxval = 2600.0
    normalize = False

    encoder = FrequencyEncoder(frequencyCutoff, numFrequencyBins, freqBinN,
                               freqBinW, minval, maxval, normalize=normalize)


    x = np.linspace(0, 100, 100)

    print '== Input signal: only zeros =='
    inputData = np.zeros(len(x))
    encoding = encoder.encode(inputData)
    pprint(encoding, numFrequencyBins, freqBinN)


    print '== Input signal: sine wave (5 Hz) =='
    f = 5
    inputData = np.sin(x * 2 * np.pi * f)
    encoding = encoder.encode(inputData)
    pprint(encoding, numFrequencyBins, freqBinN)


    print '== Input signal: 0.5 * sine wave (5 Hz) =='
    f = 5
    inputData = 0.5 * np.sin(x * 2 * np.pi * f)
    encoding = encoder.encode(inputData)
    pprint(encoding, numFrequencyBins, freqBinN)


    print '== Input signal: sine wave (10 Hz) =='
    f = 10
    inputData = np.sin(x * 2 * np.pi * f)
    encoding = encoder.encode(inputData)
    pprint(encoding, numFrequencyBins, freqBinN)

    print '== Input signal: sine wave (5 Hz) + sine wave (10 Hz) =='
    f = 5
    inputData = np.sin(x * 2 * np.pi * f) + np.sin(x * 2 * np.pi * 2 * f)
    encoding = encoder.encode(inputData)
    pprint(encoding, numFrequencyBins, freqBinN)


    print '== Input signal: 0.5 * sine wave (5 Hz) + sine wave (10 Hz) =='
    f = 5
    inputData = 0.5 * np.sin(x * 2 * np.pi * f) + np.sin(x * 2 * np.pi * 2 * f)
    encoding = encoder.encode(inputData)
    pprint(encoding, numFrequencyBins, freqBinN)

