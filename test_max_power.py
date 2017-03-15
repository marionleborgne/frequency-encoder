"""
To use ScalarEncoder on the power spectra of unpredictable sound chunks, we
need an estimate of the maximum possible value that will occur in any of
these power spectra.

The exact values of the power spectrum depends on several parameters,
including the size of the time chunk and the sampling frequency. For all
phonemes used in the Phoneme data set from UCR, sampling frequency is
(roughly) 22050 Hz, so we ignore this. This leaves the question of how
max(power) varies with size of time chunk.

We simply run through several plausible values of the chunk size,
from 32 to 1024 (the phonemes all have size 1024 samples).

Perhaps unsurprisingly, the maximum value of the raw power spectrum varies grows
with time chunk size. In addition, it is badly distributed (long tail,
high density near zero)

Both problems are solved by taking the log of the power, which is also
biologically defensible. Now max power never goes beyond 13, with a nice
bell-curve shape (not sure it's Gaussian) centered around values ranging from
6 to 10.
"""
import numpy as np
import matplotlib.pyplot as plt

data_path = 'UCR_TS_Archive_2015/Phoneme/Phoneme_TEST'
full_data = np.loadtxt(data_path, delimiter=',')
full_data = full_data[:, 1:]  # We don't need the class info for this analysis

time_chunk_sizes = (32, 64, 128, 256, 512, 1024)
plt.figure(figsize=(14, 10))
ax = [plt.subplot(3, 3, i) for i in range(1, len(time_chunk_sizes) + 1)]
print 'Analyzing data ...'
for time_chunk_size in time_chunk_sizes:
  nb_chunks = full_data.shape[1] / time_chunk_size
  new_nb_cols = nb_chunks * time_chunk_size
  t_small = full_data[:, :new_nb_cols]
  t_small = t_small.reshape(full_data.shape[0] * nb_chunks, time_chunk_size)
  psd_maximums = []
  for n in range(t_small.shape[0]):
    psd = np.log(1.0 + abs(np.fft.rfft(t_small[n, :])) ** 2)
    psd_maximums.append(np.max(psd))
  bins = np.linspace(0, np.max(psd_maximums), 100)
  plot_idx = time_chunk_sizes.index(time_chunk_size)
  ax[plot_idx].hist(psd_maximums, bins=bins)
  ax[plot_idx].set_title(
    'Chunk size: %s\n'
    'Maximum across all power spectra: %.2f' % (time_chunk_size,
                                                np.max(psd_maximums)))
  ax[plot_idx].set_ylabel('Frequency')
  ax[plot_idx].set_xlabel('Max values')

plt.tight_layout()
file_path = 'psd_maximums_distribution.png'
plt.savefig(file_path)
print 'Figure saved:', file_path
