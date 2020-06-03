import numpy as np
from diffmap import util
# ---- Data analysis


def peak_amplitudes(data, peak_list, grid_bool):
    """
    This function takes 2D data, with coordinates
    data[spectrum_number][column][time_point], and returns a list of summed
    amplitudes for a grid of 9 points centered around the peak of interest.
    Ordered as they were given in peak_list
    """
    # make sure the data are in arrays
    data = np.array(data)
    peak_list = np.array(peak_list)

    Npeaks = peak_list.shape[0]
    if len(data.shape) == 2:
        Nspectra = 1
    else:
        Nspectra = data.shape[0]

    # initialize the data shape and amp_list to output, and the phase
    # correction...wait, amp_list isn't used anywhere
    N1 = data.shape[-1]
    # amp_list = peak_list.copy()
    pcorr_f1 = util.get_phasecorr(N1, 1)

    # we output a peak for each 2d spectrum: peak_amp[spectrum, peak]
    if Nspectra == 1:
        peak_gridamps = np.full((Npeaks), np.nan)
    else:
        peak_gridamps = np.full((Nspectra, Npeaks), np.nan)

    # fill out the grid of data: each spectrum, each peak
    for spectrum in np.arange(Nspectra):
        temp_data = data[spectrum, :, :]
        temp_spec = np.fft.fftshift(np.fft.fft(temp_data, axis=1), axes=1)
        temp_spec_r = (temp_spec * pcorr_f1).real

        for peak_num in np.arange(Npeaks):
            peak_row = int(peak_list[peak_num, 0])
            peak_col = int(peak_list[peak_num, 1])

            amp = np.sum(temp_spec_r[peak_row - grid_bool:
                                     peak_row + grid_bool + 1,
                                     peak_col - grid_bool:
                                     peak_col + grid_bool + 1])

            if Nspectra == 1:
                peak_gridamps[peak_num] = amp
            else:
                peak_gridamps[spectrum, peak_num] = amp

    return peak_gridamps
