from frequency_encoder import FrequencyEncoder
import numpy as np
import unittest



class TestStringMethods(unittest.TestCase):
  def setUp(self):
    self.numFrequencyBins = 5
    self.freqBinN = 5
    self.freqBinW = 1
    self.minval = 0.0
    self.maxval = 7.9

    self.encoder = FrequencyEncoder(self.numFrequencyBins,
                                    self.freqBinN,
                                    self.freqBinW,
                                    minval=self.minval,
                                    maxval=self.maxval)

    self.x = np.linspace(0, 100, 100)


  def test_zeros(self):
    inputData = np.zeros(len(self.x))
    encoding = self.encoder.encode(inputData)
    expectedEncoding = np.zeros((self.numFrequencyBins, self.freqBinN),
                                dtype=int)
    expectedEncoding[0, 0] = 1
    expectedEncoding[1, 0] = 1
    expectedEncoding[2, 0] = 1
    expectedEncoding[3, 0] = 1
    expectedEncoding[4, 0] = 1
    self.assertSequenceEqual(list(encoding),
                             list(expectedEncoding.flatten()))


  def test_sine_5Hz(self):
    f = 5
    inputData = np.sin(self.x * 2 * np.pi * f)
    encoding = self.encoder.encode(inputData)
    expectedEncoding = np.zeros((self.numFrequencyBins, self.freqBinN),
                                dtype=int)
    expectedEncoding[0, 4] = 1
    expectedEncoding[1, 0] = 1
    expectedEncoding[2, 0] = 1
    expectedEncoding[3, 0] = 1
    expectedEncoding[4, 0] = 1
    self.assertSequenceEqual(list(encoding),
                             list(expectedEncoding.flatten()))


  def test_half_sine_5Hz(self):
    f = 5
    inputData = 0.5 * np.sin(self.x * 2 * np.pi * f)
    encoding = self.encoder.encode(inputData)
    expectedEncoding = np.zeros((self.numFrequencyBins, self.freqBinN),
                                dtype=int)
    expectedEncoding[0, 3] = 1
    expectedEncoding[1, 0] = 1
    expectedEncoding[2, 0] = 1
    expectedEncoding[3, 0] = 1
    expectedEncoding[4, 0] = 1
    self.assertSequenceEqual(list(encoding),
                             list(expectedEncoding.flatten()))


  def test_sine_10Hz(self):
    f = 10
    inputData = np.sin(self.x * 2 * np.pi * f)
    encoding = self.encoder.encode(inputData)
    expectedEncoding = np.zeros((self.numFrequencyBins, self.freqBinN),
                                dtype=int)
    expectedEncoding[0, 1] = 1
    expectedEncoding[1, 4] = 1
    expectedEncoding[2, 0] = 1
    expectedEncoding[3, 0] = 1
    expectedEncoding[4, 0] = 1
    self.assertSequenceEqual(list(encoding),
                             list(expectedEncoding.flatten()))


  def test_sine_5Hz_and_sine_10Hz(self):
    f = 5
    inputData = np.sin(self.x * 2 * np.pi * f) + np.sin(
      self.x * 2 * np.pi * 2 * f)
    encoding = self.encoder.encode(inputData)
    expectedEncoding = np.zeros((self.numFrequencyBins, self.freqBinN),
                                dtype=int)
    expectedEncoding[0, 4] = 1
    expectedEncoding[1, 4] = 1
    expectedEncoding[2, 0] = 1
    expectedEncoding[3, 0] = 1
    expectedEncoding[4, 0] = 1
    self.assertSequenceEqual(list(encoding),
                             list(expectedEncoding.flatten()))


  def test_half_sine_5Hz_and_sine_10Hz(self):
    f = 5
    inputData = 0.5 * np.sin(self.x * 2 * np.pi * f) + np.sin(
      self.x * 2 * np.pi * 2 * f)
    encoding = self.encoder.encode(inputData)
    expectedEncoding = np.zeros((self.numFrequencyBins, self.freqBinN),
                                dtype=int)
    expectedEncoding[0, 3] = 1
    expectedEncoding[1, 4] = 1
    expectedEncoding[2, 0] = 1
    expectedEncoding[3, 0] = 1
    expectedEncoding[4, 0] = 1
    self.assertSequenceEqual(list(encoding),
                             list(expectedEncoding.flatten()))



if __name__ == '__main__':
  unittest.main()
