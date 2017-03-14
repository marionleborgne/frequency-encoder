# To use ScalarEncoder on the power spectra of unpredictable sound chunks, we
# need an estimate of the maximum possible value that will occur in any of
# these power spectra.

# The exact values of the power spectrum depends on several parameters,
# including the size of the time chunk and the sampling frequency. For all
# phonemes used in the Phoneme data set from UCR, sampling frequency is
# (roughly) 22050 Hz, so we ignore this. This leaves the question of how
# max(power) varies with size of time chunk.

# We simply run through several plausible values of the chunking size, from 32 to 1024 (the phonemes all have size 1024 samples).

# Perhaps unsurprisingly, the maximum value of the raw power spectrum varies grows
# with time chunk size. In addition, it is badly
# distributed (long tail, high density near zero)

# Both problems are solved by taking the log of the power, which is also
# biologically defensible. Now max power never goes beyond 13, with a nice
# bell-curve shape (not sure it's Gaussian) centered around values ranging from
# 6 to 10.


fulldata = np.loadtxt('./Phoneme_TEST', delimiter=',')
fulldata = fulldata[:,1:]  # We don't need class info for this
print fulldata.shape
timechunksizes = (32, 64, 128, 256, 512, 1024)
for tt in timechunksizes:
    timechunk = tt
    nbchunks = np.floor(fulldata.shape[1] / timechunk).astype(int) 
    newnbcols = nbchunks * timechunk
    tsmallrs = fulldata[:,:newnbcols]
    #print tsmallrs.shape
    tsmallrs = tsmallrs.reshape(fulldata.shape[0]*nbchunks, timechunk)
    #print tsmallrs.shape
    maxes = []
    for n in range(tsmallrs.shape[0]):
        #mypsd = abs(np.fft.rfft(tsmallrs[n,:]))**2             # Max raw power can vary by orders of magnitude 
        mypsd = np.log(1.0+abs(np.fft.rfft(tsmallrs[n,:]))**2)  # Log is much more predictable 
        maxes.append(np.max(mypsd))
    print "Maximum value across all power spectra for time chunk size", timechunk, ": "
    print np.max(maxes)
    mbins = np.linspace(0, np.max(maxes), 100)
    #h = np.histogram(maxes, bins=mbins)
    plt.hist(maxes, bins=mbins)
    plt.show()
    
