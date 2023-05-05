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

    freq = spectrum.values[:, 0].copy()
    intens = spectrum.values[:, 1].copy()

    new_freq = np.linspace(start = freq[0], stop = freq[-1], num = length, endpoint = True)
    new_intens = []

    j = 1
    for f in range(length):         # each new interpolated intensity
        interpolated_int = 0
        
        if f == 0:
            new_intens.append(intens[0])
            continue

        if f == length - 1:
            new_intens.append(intens[length - 1])
            continue

        for i in range(j, length):     # we compare with "each" measured intensity

            if new_freq[f] <= freq[i]: # so for small i that's false. That's true for the first freq[i] greater than new_freq[f]
                
                interpolated_int = linear_inter(freq[i - 1], freq[i], new_freq[f], intens[i-1], intens[i])
                break

            else:
                j += 1 # j is chosen in such a way, because for every new "freq" it will be greater than previous

        new_intens.append(interpolated_int)

    data = [[new_freq[i], new_intens[i]] for i in range(length)]
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
    from scipy.fft import fft, fftfreq, fftshift

    freq = spectrum.values[:, 0]
    intens = spectrum.values[:, 1]
    spacing = np.mean(np.diff(freq))

    # Fourier Transform

    if absolute:
        FT_intens = np.abs(fftshift(fft(intens, norm = "ortho")))
    else:
        FT_intens = fftshift(fft(intens, norm = "ortho"))

    time = fftshift(fftfreq(len(freq), spacing))

    data = np.transpose(np.stack((time, FT_intens)))

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
    from scipy.fft import ifft, fftfreq, fftshift

    time = spectrum.values[:, 0]
    intens = spectrum.values[:, 1]
    spacing = np.mean(np.diff(time))

    # Fourier Transform

    if absolute:
        FT_intens = np.abs(ifft(intens, norm = "ortho"))
    else:
        FT_intens = ifft(intens, norm = "ortho")
    freq = fftshift(fftfreq(len(time), spacing))

    # just sort data to make slicing later possible
    '''
    data = [[freq[i], FT_intens[i]] for i in range(len(freq))]
    data.sort()
    '''
    data = np.transpose(np.stack((freq, FT_intens)))

    return pd.DataFrame(data)

def cut(spectrum, start, end, how = "units"):
    '''
    Returns the \"spectrum\" limited to the borders. 

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y-axis.

    start - start of the segment, to which the spectrum is limited.

    end - end of the segment, to which the spectrum is limited.

    how - defines meaning of \"start\" and \"end\". If \"units\", then those are values on X-axis. If \"fraction\", then the fraction of length of X-axis. If \"index\", then corresponding indices of border observations.
    
    RETURNS:

    Limited spectrum DataFrame.

    '''

    import pandas as pd
    import numpy as np
    from math import floor

    if how == "units":
        s = np.searchsorted(spectrum.values[:, 0], start)
        e = np.searchsorted(spectrum.values[:, 0], end)
        cut_spectrum = pd.DataFrame(spectrum.values[s: e, :])

    elif how == "fraction":
        s = floor(start*spectrum.shape[0])
        e = floor(end*spectrum.shape[0])
        cut_spectrum = pd.DataFrame(spectrum.values[s:e, :])

    elif how == "index":
        cut_spectrum = pd.DataFrame(spectrum.values[start:end, :]) 

    else:
        raise ValueError("Argument not defined.")

    return cut_spectrum

def plot(spectrum, color = "darkviolet", title = "Spectrum", type = "wl", what_to_plot = "abs", min = 0, max = 0):
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
    from math import floor, log10

    spectrum_safe = spectrum.copy()
    
    # simple function to round to significant digits

    def round_to_dig(x, n):
        return round(x, -int(floor(log10(abs(x)))) + n - 1)

    # invalid arguments

    if type not in ("wl", "freq", "time"):
        raise Exception("Input \"type\" not defined.")
    
    if what_to_plot not in ("abs", "imag", "real"):
        raise Exception("Input \"what_to_plot\" not defined.")

    # if we dont want to plot whole spectrum

    n_points = len(spectrum_safe.values[:, 0])

    to_cut = min != max
    if to_cut:

        inf = round(np.real(spectrum_safe.values[0, 0].copy()))
        sup = round(np.real(spectrum_safe.values[-1, 0].copy()))

        s = np.searchsorted(spectrum_safe.values[:, 0], min)
        e = np.searchsorted(spectrum_safe.values[:, 0], max)
        spectrum_safe = pd.DataFrame(spectrum_safe.values[s: e, :])

    values = spectrum_safe.values[:, 1].copy()
    X = spectrum_safe.values[:, 0].copy()
    
    # what do we want to have on Y axis

    if what_to_plot == "abs":
        values = np.abs(values)
    if what_to_plot == "real":
        values = np.real(values)
    if what_to_plot == "imag":
        values = np.imag(values)

    # start to plot

    f, ax = plt.subplots()
    plt.plot(X, values, color = color)
    plt.grid()
    if to_cut:
        plt.title(title + " [only part shown]")
    else:
        plt.title(title)
    plt.ylabel("Intensity")    
    if type == "wl":
        plt.xlabel("Wavelength [nm]")
        unit = "nm"
    if type == "freq":
        plt.xlabel("Frequency [THz]")
        unit = "THz"
    if type == "time":
        plt.xlabel("Time [ps]")
        unit = "ps"

    # quick stats

    spacing = round_to_dig(np.mean(np.diff(np.real(X))), 3)
    p_per_unit = floor(1/np.mean(np.diff(np.real(X))))

    if to_cut:
        plt.text(1.05, 0.85, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) + "\nFull X-axis range: {} - {} ".format(inf, sup) + unit , transform = ax.transAxes)
    else:
        plt.text(1.05, 0.9, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) , transform = ax.transAxes)

    plt.show()

def shift(spectrum, shift):
    '''
    Shifts the spectrum by X axis. Warning: only values on X axis are modified.

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y axis.
    
    shift - size of a shift in X axis units.

    RETURNS:

    Shifted spectrum DataFrame.
    '''
    import pandas as pd
    import numpy as np

    data = spectrum.values
    new_data = data.copy()
    new_data[:, 0] = new_data[:, 0] + shift

    return pd.DataFrame(new_data)

def zero_padding(spectrum, how_much):
    '''
    Add zeros on Y-axis to the left and right of data with constant (mean) spacing on X-axis.
    '''
    import pandas as pd
    import numpy as np
    from math import floor

    length = floor(how_much*spectrum.shape[0])
    spacing = np.mean(np.diff(spectrum.values[:, 0]))
    
    left_start = spectrum.values[0, 0] - spacing*length
    left = np.linspace(left_start, spectrum.values[0, 0], endpoint = False, num = length - 1)
    left_arr = np.transpose(np.stack((left, np.zeros(length-1))))

    right_end = spectrum.values[-1, 0] + spacing*length
    right = np.linspace(spectrum.values[-1, 0] + spacing, right_end, endpoint = True, num = length - 1)
    right_arr = np.transpose(np.stack((right, np.zeros(length-1))))

    arr_with_zeros = np.concatenate([left_arr, spectrum.values, right_arr])

    return pd.DataFrame(arr_with_zeros)

def replace_with_zeros(spectrum, start, end):
    '''
    description
    '''

    import numpy as np
    import pandas as pd

    data = spectrum.values.copy()
    if start == "min":
        s = 0
    else:
        s = np.searchsorted(spectrum.values[:, 0], start)

    if end == "max":
        e = spectrum.shape[0]
    else:
        e = np.searchsorted(spectrum.values[:, 0], end)

    data[s: e, 1] = np.zeros(e-s)

    return pd.DataFrame(data)

def smart_shift(spectrum, shift):
    '''
    Shift spectrum by X axis. Values on X axis are NOT modified. Some of the values on Y axis are deleted, \"empty\" values are zero by default.
    '''

    import numpy as np
    import pandas as pd
    from math import floor

    data = spectrum.values.copy()
    spacing = np.mean(np.diff(data[:, 0]))
    index_shift = floor(shift/np.real(spacing))

    # why doesnt it work?
    ''' 
    if index_shift >= 0:
        data[-index_shift:, 1] = np.zeros(data[-index_shift:, 1].shape)
    if index_shift <= 0:
        data[:-index_shift, 1] = np.zeros(data[:-index_shift, 1].shape)
    '''
    data[:, 1] = np.roll(data[:, 1], index_shift)
        
    return pd.DataFrame(data)