# A library containing functions that I personally use (and of course wrote) to analyze optical spectra of laser pulses measured by 
# Optical Spectrum Analyzer (OSA). Large part of the functions is focused on interference patterns analysis - usually some kind of fringes.
# Spectra are read from .tsv files and contain two columns: "Wavelength" and corresponding spectral "Intensity".
# The library is expected to be hugely extended.

def find_period(spectrum, height = 1, hist = False):
    '''
    Function finds period in interference fringes by looking for wavelengths, where intensity is around given height and is decreasing. 

    ARGUMENTS:

    spectrum - DataFrame with Wavelength\Frequency in first column and Intensity in second.

    height - height, at which we are looking for negative slope. Height is the fraction of intensity mean.

    hist - if to plot the histogram of all found periods.

    RETURNS:

    (mean, std) - mean and standard deviation of all found periods. 
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    wl = spectrum.values[:, 0]
    intens = spectrum.values[:, 1]
    h = height * np.mean(intens)
    nodes = []

    for i in range(2, len(intens) - 2):
        # decreasing
        if intens[i-2] > intens[i-1] > h > intens[i] > intens[i+1] > intens[i+2]:
            nodes.append(wl[i])

    diff = np.diff(np.array(nodes))

    if hist:
        plt.hist(diff, color = "orange")
        plt.xlabel("Period length [nm]")
        plt.ylabel("Counts")
        plt.show()

    mean = np.mean(diff)
    std = np.std(diff)
    diff_cut = [d for d in diff if np.abs(d - mean) < std]
    
    return np.mean(diff_cut), np.std(diff_cut)

def constant_spacing(spectrum):
    '''
    Transformation of a spectrum to have constant spacing on X-axis (wavelength) by linearly interpolating two nearest values on 
    Y-axis (intensity).

    ARGUMENTS:

    spectrum - DataFrame with Wavelength/Frequency on X axis and Intensity on Y axis.

    RETURNS:

    Spectrum DataFrame with Wavelength/Frequency on X axis and Intensity on Y axis AND constant spacing on X axis.
    '''

    import numpy as np
    import pandas as pd

    def linear_inter(a, b, c, f_a, f_b):
        if a == b:
            return (f_a + f_b)/2
        
        else:
            f_c =  f_a * np.abs(b-c) / np.abs(b-a) + f_b * np.abs(a-c) / np.abs(b-a)
            return f_c

    length = spectrum.shape[0]

    freq = spectrum.values[:, 0]
    int = spectrum.values[:, 1]

    new_freq = np.linspace(start = freq[0], stop = freq[-1], num = length, endpoint = True)
    new_int = []

    j = 0
    for f in range(length):         # each new interpolated intensity
        interpolated_int = 0

        for i in range(j, length):     # we compare with "each" measured intensity

            if new_freq[f] < freq[i]:
                if i < length - 1:
                    next_freq = freq[i+1]
                    next_int = int[i+1]

                else:
                    next_freq = freq[i]
                    next_int = int[i] 
                
                interpolated_int = linear_inter(freq[i], next_freq, new_freq[f], int[i], next_int)
                break

            else:
                i += 1
                j += 1

        new_int.append(interpolated_int)

    data = [[new_freq[i], new_int[i]] for i in range(length)]
    df_spectrum = pd.DataFrame(data)

    return df_spectrum

def make_it_visible(spectrum, segment_length):
    '''
    The function improves visibility of interference fringes by subtracting local minima of spectrum. Each local minimum is a minimum of a
    segment of segment_length points. Last segment might be shorter.

    ARGUMENTS:

    spectrum - DataFrame with Wavelength/Frequency on X axis and Intensity on Y axis.

    segment_length - length of a segment of which we take a local minimum. It needs to be chosen manually and carefully as it highly
    influences quality of visibility.

    RETURNS:

    Spectrum DataFrame spectrum with subtracted offset and increased visibility
    '''

    import numpy as np
    import pandas as pd
    from math import floor as flr

    wl = spectrum.values[:, 0]
    intens = spectrum.values[:, 1]

    minima = []
    samples_num = len(intens) // segment_length + 1

    # find "local" minima

    for i in range(samples_num):
        start = segment_length*i
        end = segment_length*(i+1)

        if start >= len(intens): break

        if end > len(intens) - 1:
            end = len(intens)

        minimum = np.min(intens[start: end])
        minima.append(minimum)

    # subtract the minima

    new_intens = []
    for i in range(len(intens)):
        new =  intens[i] - minima[flr(i/segment_length)]
        new_intens.append(new)

    # and return a nice dataframe

    return pd.DataFrame(np.array([[wl[i], new_intens[i]] for i in range(len(wl))]))

def wl_to_freq(spectrum):
    '''
    Transformation from wavelength domain [nm] to frequency domain [THz].

    ARGUMENTS:

    spectrum - DataFrame with Wavelength [nm] on X axis and Intensity on Y axis.

    RETURNS:

    Spectrum DataFrame with Frequency [THz] on X axis and Intensity on Y axis.
    '''

    import numpy as np
    import pandas as pd

    c = 299792458 # light speed

    FREQ = np.flip(c / spectrum.values[:, 0] / 1e3) # output in THz
    INTENS = np.flip(spectrum.values[:, 1])
    data = [[FREQ[i], INTENS[i]] for i in range(len(FREQ))]

    return pd.DataFrame(data)

def fourier(spectrum, absolute = False):
    '''
    Performs Fourier Transform (usually from \"frequency\" domain to \"time\" domain).

    ARGUMENTS:

    spectrum - DataFrame with Frequency on X axis and Intensity on Y axis.
        
    absolute - if \"True\", then absolute intensities are returned.

    RETURNS:

    Fourier Transformed spectrum DataFrame.
    '''

    import numpy as np
    import pandas as pd
    from scipy.fft import fft, fftfreq

    freq = spectrum.values[:, 0]
    intens = spectrum.values[:, 1]
    spacing = np.mean(np.diff(freq))

    # Fourier Transform

    if absolute:
        FT_intens = np.abs(fft(intens))
    else:
        FT_intens = fft(intens)

    time = fftfreq(len(freq), spacing)

    # just sort data to make slicing later possible

    data = [[time[i], FT_intens[i]] for i in range(len(time))]
    data.sort()

    return pd.DataFrame(data)

def inv_fourier(spectrum, absolute = False):
    '''
    Performs Inverse Fourier Transform (usually from \"time\" domain to \"frequency\" domain).
    
    ARGUMENTS:

    spectrum - DataFrame with Time on X axis and Intensity on Y axis.
    
    absolute - if \"True\", then absolute intensities are returned.

    RETURNS:

    Fourier Transformed spectrum DataFrame.
    '''

    import numpy as np
    import pandas as pd
    from scipy.fft import ifft, fftfreq

    time = spectrum.values[:, 0]
    intens = spectrum.values[:, 1]
    spacing = np.mean(np.diff(time))

    # Fourier Transform

    if absolute:
        FT_intens = np.abs(ifft(intens))
    else:
        FT_intens = ifft(intens)
    freq = fftfreq(len(time), spacing)

    # just sort data to make slicing later possible

    data = [[freq[i], FT_intens[i]] for i in range(len(freq))]
    data.sort()

    return pd.DataFrame(data)

def cut(spectrum, start, end, percentage = False):
    '''
    Returns the \"spectrum\" limited to the borders.

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y-axis.

    start - start of the segment, to which the spectrum is limited.

    end - end of the segment, to which the spectrum is limited.

    percentage - if \"False\" then \"start\" and \"end\" mean the indices of border observations. Otherwise they are in [0, 1] and the border observations' indices are \"start*len(spectrum)\" and \"end*len(spectrum)\".
    
    RETURNS:

    Limited spectrum DataFrame.

    '''

    import pandas as pd
    import numpy as np

    cut_spectrum = pd.DataFrame(spectrum.values[start:end, :]) 

    return cut_spectrum

def plot(spectrum, color = "darkviolet", title = "Spectrum", type = "wl"):
    '''
    Fast spectrum plotting using matplotlib.pyplot library.

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y axis.

    color - color of the plot.

    title - title of the plot.

    type - \"wl\" (Wavelength) or \"freq\" (Frequency) or \"time\" (Time). Determines the label of X-axis.

    RETURNS:

    Continuous plot of the \"spectrum\"/.
    '''

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if type not in ("wl", "freq", "time"):
        raise Exception("Type not defined.")

    plt.plot(spectrum.values[:, 0], spectrum.values[:, 1], color = color)
    plt.grid()
    plt.title(title)
    if type == "wl":
        plt.xlabel("Wavelength [nm]")
    if type == "freq":
        plt.xlabel("Frequency [THz]")
    if type == "time":
        plt.xlabel("Time [ps]")
    plt.ylabel("Intensity")
    plt.show()

def shift(spectrum, shift):
    '''
    Shifts the spectrum on X axis.

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y axis.
    
    shift - size of a shift in X axis units.

    RETURNS:

    Shifted spectrum DataFrame.
    '''
    import pandas as pd
    import numpy as np

    data = spectrum.values
    data[:, 0] = data[:, 0] + shift

    return pd.DataFrame(data)