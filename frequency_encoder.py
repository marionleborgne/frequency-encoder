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
    def __init__(self, numFrequencyBins, freqBinN, freqBinW, minval, maxval):

        self.numFrequencyBins = numFrequencyBins
        self.freqBinN = freqBinN
        self.freqBinW = freqBinW
        self.minval = minval
        self.maxval = maxval

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
        Encodes inputData and puts the encoded value into the numpy output
        array, which is a 1-D array of length returned by getWidth().

        :param inputData: (np.array) Data to encode.
        :param output: (np.array) 1-D array. Encoder output.
        """

        if type(inputData) != np.ndarray:
            raise TypeError(
                'Expected a numpy array but input type is %s' % type(inputData))

        if inputData is SENTINEL_VALUE_FOR_MISSING_DATA:
            output[0:self.outputWidth] = 0
        else:
            freqs = getFreqs(inputData)
            freqBinSize = len(freqs) / self.numFrequencyBins
            binEncodings = []
            for i in range(self.numFrequencyBins):
                freqBin = freqs[i * freqBinSize:(i + 1) * freqBinSize]
                binVal = np.max(freqBin)
                binEncoding = self.scalarEncoder.encode(binVal)
                binEncodings.append(binEncoding.tolist())

            output[0:self.outputWidth] = np.array(binEncodings).flatten()


def getFreqs(data, log=True):
    fft = abs(np.fft.rfft(data)) ** 2
    if log:
        return np.log(1 + fft)
    else:
        return fft


def pprint(encoding, numFrequencyBins, freqBinSize):
    for i in range(numFrequencyBins):
        print ('Freq bin %s: '
               '%s' % (i, encoding[i * freqBinSize:(i + 1) * freqBinSize]))
