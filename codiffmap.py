# -- coDiffMap
# -- May 13, 2019
# -- Core Python DiffMap functionality written by ROBERT BLUM
# -- coDiffMap adaptation written by JARED ROVNY

# -- processing
# 1. fft_phase_1d
# 2. get_phasecorr

# -- preparation
# 3. get_interesting_columns
# 4. get_interesting_mask_columns
# 5. get_interesting_columns_all2D

# -- sampling
# 6. sampling_mask
# 7. stagger_sample
# 8. sparsify_staggered

# -- coDiffMap core
# 9. setup_and_run_codiffmap
# 10. run_codiffmap_1D
# 11. make_push_points
# 12. p1_proj
# 13. p2_proj_star
# 14. difference_map_star

# -- analysis
# 15. peak_amplitudes


import numpy as np
import copy as copy


# ---- data processing

def fft_phase_1d(origdata_2d):
    """
    This takes 2D data, and applies an fft and phase correction along 1
    dimension only. The output is the "semi-FFT'd" data, from which we can take
    individual 1D slices along t1.
    """
    N2, N1 = origdata_2d.shape
    row, col = np.indices(origdata_2d.shape)
    n2, n1 = row - N2 / 2, col - N1 / 2

    # define shift amounts, with optional offset by 1/2 for the indirect
    # dimension
    t_shift_1 = (N1 / 2 - 1 / 2)
    t_shift_2 = (N2 / 2)

    # get frequencies in terms of point number
    f_1 = n1 / N1
    f_2 = n2 / N2

    # make phase correction matrices pcorrmat =
    # np.exp(-1.j*2*np.pi*f_2*t_shift_2)*np.exp(-1.j*2*np.pi*f_1*t_shift_1)
    pcorrmat_f2 = np.exp(-1.j * 2 * np.pi * f_2 * t_shift_2)

    # origdata_fft = np.fft.fftshift(np.fft.fft2(origdata_2d))
    # origdata_fftr = (origdata_fft*pcorrmat).real

    origdata_semifft = np.fft.fftshift(np.fft.fft(origdata_2d, axis=0), axes=0)
    # origdata_semifft = np.fft.fft(origdata_2d,axis=0)
    origdata_semifft_ph = origdata_semifft * pcorrmat_f2

    return origdata_semifft_ph


def get_phasecorr(N, offbool):
    """
    Creates a phase correction array, to make the FFT of a Hermitian
    array all real. **Only works in 1D right now!**
    'N' is the length of the vector
    'offbool' is a boolean that controls whether there's a dt/2 shift
    in the time values:
    - if time[N/2] is t = 0, set offbool = false
    - if time[N/2] is t = dt/2, set offbool = true

    Returns a 1D vector 'phasecorr' of length N.

    This function assumes an even number of points, so it's yet not as general
    as it could be.
    """
    # phasecorr = np.exp(-1j*2*np.pi*(N/2-0.5*offbool)/N*(N/2-np.arange(N)))

    return np.exp(-1j * 2 * np.pi * (N / 2 - 0.5 * offbool) /
                  N * (N / 2 - np.arange(N)))


# ---- data preparation

def get_interesting_columns(datapath, column_indices, spectrum_index, n_rows):
    import numpy as np
    """
    Load "interesting" columns of data from a given 2d data set.

    This function is designed to keep only the relevant columns in memory,
    since we need a lot of local data for the projection method.
    """

    data_source = str(datapath) + "_" + str(spectrum_index) + ".txt"

    real_cols = list(range(0, n_rows * 2, 2))
    imag_cols = list(range(1, n_rows * 2, 2))
    origdata_2d = np.loadtxt(data_source, usecols=real_cols) \
        - 1.j * np.loadtxt(data_source, usecols=imag_cols)

    origdata_2d_fft_ph = fft_phase_1d(origdata_2d)

    compressed_data = origdata_2d_fft_ph[column_indices, :]
    return compressed_data


def get_interesting_mask_columns(mask, column_indices):
    """
    Load "interesting" mask columns. Very simple function.
    """

    compressed_mask = mask[column_indices, :]
    return compressed_mask


def get_interesting_columns_all2D(datapath, column_indices, number_of_spectra,
                                  n_rows):
    """
    This will return a LIST of 2D DATA SETS. Each 2D spectrum in the list
    will just be from the columns chosen by the user.
    """

    n_columns = len(column_indices)

    all_relevant_data = np.zeros((number_of_spectra, n_columns, n_rows),
                                 dtype=np.complex)
    for spectrum_index in range(1, number_of_spectra + 1):
        new_data = get_interesting_columns(datapath, column_indices,
                                           spectrum_index, n_rows)
        all_relevant_data[spectrum_index - 1] = new_data
    return all_relevant_data


# ---- coDiffMap functions


def setup_and_run_codiffmap(data_in, Nsparse, mask, offbool, push_param,
                            N_iter):
    """
    Shortcut to set up the required input data and run the diffmap function...
    Note: mask needs to correspond to "data_in," column by column.
    """
    Ndense = np.int(len(data_in[0][0]) / 2)
    N3D = np.int(len(data_in))

    data_gaps, meas_points, push_points = sparsify_staggered(data_in, Nsparse,
                                                             N3D, offbool)
    data_out = np.copy(data_in)

    for slice2d in np.arange(len(data_in)):
        for column in np.arange(len(data_in[0])):
            data_gaps_1d = data_gaps[slice2d][column].copy()
            meas_points_1d = meas_points[slice2d][column].copy()
            mask_1d = mask[column].copy()
            push_points_1d = push_points[slice2d][column].copy()

            data_out_1d = run_codiffmap_1D(data_gaps_1d, N_iter,
                                           meas_points_1d, mask_1d,
                                           push_points_1d, push_param)
            data_out[slice2d][column] = np.asarray(data_out_1d.copy())

    return data_out


def run_codiffmap_1D(data_in, N_iter, measured_points, mask,
                     push_points, push_param):
    """
    Runs the difference map on data_in, with N_iter iterations.
    Outputs P2(D*(data_in)^N); that is; it applies P2 at the end of the
    iteration loop.

    TODO: maybe change the inputs so that measured_points is generated from
          data_in? Might need to input a row mask or an Nsparse value instead.

    """
    data_out = np.copy(data_in)

    # infer offbool from the form of the given data...may not be stable?
    # update: it's not stable, alas
    # N = len(data_in)
    # offbool = data_in.real[int(N/2)] == data_in.real[int(N/2 - 1)]
    offbool = 1
    # do the difference map N_iter times
    for n in np.arange(N_iter):
        data_out = difference_map_star(data_out, measured_points, mask,
                                       offbool, push_points, push_param)

    # apply final P2* projection, to restore the measured data points Note: for
    # the difference_map_star, it is important that this last step is a P2*,
    # not P2!
    data_out = p2_proj_star(data_out, measured_points, push_points, push_param)

    return data_out


def make_push_points(data, stagger_sampling_mask):
    """
    This function will take your data (many 2D slices), and a staggered
    sampling schedule, and make a data array of "push" points, f2-column by
    f2-column. For a given 2D slice, these push points will be:
    -- at measured t1 points: =NaN
    -- at unmeasured t1 points: is equal to the data at the
    same t1, but from a different (the closest available) 2D slice.
    """

    n_slices, n_columns, n_rows = data.shape

    push_points = np.zeros((n_slices, n_columns, n_rows), dtype=np.complex)
    push_points_1d = np.zeros(n_rows, dtype=np.complex)

    # walk through the data column-by-column
    for column in np.arange(n_columns):

        # walk through the data spectrum-by-spectrum
        for spectrum in np.arange(n_slices):
            # first, initialize: push_points_1d is NaN for sampled points
            push_points_1d = [np.nan if x == 1 else 0 for x in
                              stagger_sampling_mask[spectrum]]

            # now, for this particular {column,spectrum},
            # and FOR EACH POINT IN PUSH_POINTS, walk through the other
            # spectra and find a suitable data point
            for i in np.arange(n_rows):
                if(np.isnan(push_points_1d[i])):
                    continue
                else:
                    # now, walk through the spectra again: for e.g. a missing
                    # point in spectrum=3, look at spectra #4,2,5,1,...,etc.,
                    # until a point is found
                    for spec in np.arange(2 * n_slices):
                        # 1,-1,2,-2,3,-3...
                        ds = (np.floor(spec / 2) + 1) * (-1.0)**spec
                        this_spectrum = spectrum + ds
                        # make sure our spectrum index isn't out of bounds
                        if(this_spectrum < 0 or this_spectrum >= n_slices):
                            continue
                        # if the point has been sampled in this spectrum, use
                        # it, and stop looking
                        if(stagger_sampling_mask[int(this_spectrum)][i] == 1):
                            push_points_1d[i] = \
                                data[int(this_spectrum)][column][i]
                            break
                    # if there were no sampled point found, break with error.
                    else:
                        print('ERROR: no suitable data point found!')
                        break
            push_points[spectrum][column] = push_points_1d
    return np.asarray(push_points)


def p1_proj(data_t, mask, offbool):
    """
    Applies the P1 projection to a 1D array 'data_t', according to 'mask'.

    Note: since P1 needs data to be in the phased frequency domain, this
    function FFTs and phases the data, applies the mask, and then unphases and
    IFFTs the data before returning it.
    """
    # FFT input data
    data_f = np.fft.fftshift(np.fft.fft(data_t))

    # get the phase correction wave
    phasecorr = get_phasecorr(len(data_f), offbool)

    # phase correct input data and throw out the imaginary part
    data_f_ph = (data_f * phasecorr).real

    # apply p1 mask to data_f_ph
    data_f_ph *= (mask != 0)
    data_f_ph[mask == 1] *= (data_f_ph[mask == 1] > 0)
    data_f_ph[mask == -1] *= (data_f_ph[mask == -1] < 0)

    # put data back into time domain by unphasing and IFFTing
    # the data becomes complex again, of course
    p1_data_t = np.fft.ifft(np.fft.ifftshift(data_f_ph / phasecorr))

    return p1_data_t


def p2_proj_star(data_t, measured_points, push_points, push_param):
    """
    Applies the P2* projection to a 1D array 'data_t'.

    Overwrites some of the points in 'data_t' with original measured values,
    which are stored in 'measured_points'. 'measured_points' is NaN for points
    that weren't sampled, and equal to the measured values for points that
    were sampled.
    """
    measured_points = np.array(measured_points)
    push_points = np.array(push_points)

    # copy data to output array, force to be complex
    p2_data_t = np.asarray(data_t).astype(complex)

    # measured points "m", and un-measured points "p"
    m_points_list = ~np.isnan(measured_points)
    p_points_list = ~np.isnan(push_points)

    # reset measured values. For un-measured values, push towards the measured
    # values from the other spectra.
    p2_data_t[m_points_list] = measured_points[m_points_list]
    p2_data_t[p_points_list] = (1.0 - push_param) * p2_data_t[p_points_list] \
        + push_param * push_points[p_points_list]

    return p2_data_t


def difference_map_star(data_t, measured_points, mask, offbool, push_points,
                        push_param):
    """
    Applies one step of the difference map algorithm to a 1D array 'data_t'.

    Uses the functions p1_proj and p2_proj_star internally.
    D = 1 + P1 (2 P2* - 1) - P2*
    """

    # calculate (2 P2* - 1)[data_t] first, for clarity's sake
    data_temp = 2.0 * p2_proj_star(data_t, measured_points, push_points,
                                   push_param) - data_t

    d_data_t = data_t + p1_proj(data_temp, mask, offbool) \
        - p2_proj_star(data_t, measured_points, push_points, push_param)
    return d_data_t


# ---- sampling


def sampling_mask(Ndense, Nsparse, offbool, lastpoint_bool):
    """
    Makes a "row mask" array, for a given number of dense points and
    a given number of sparse points. Ndense is number of dense points
    in t >= 0! The output wave will be length (2*Ndense).

    This version of sampling_mask ensures we have sampled the ``last'',
    greatest-|t1| point.

    if offbool == True, then the central point (at t = dw/2) is reflected
    to the t < 0 side of the vector. If offbool == False, then the central
    point is at t = 0 and isn't duplicated.

    NOTE: the rounding convention in Igor is "away from zero." In numpy/python3
    it's "towards even." I'm implementing the Igor version here, but I might
    change it to the python version later. It will change the row choices in
    some cases, though!

    TODO: enforce 1 <= Nsparse <= Ndense properly
    """
    # initialize positive side as nans, not zeroes
    row_mask_pos = np.full(Ndense, np.nan)
    # row_mask_pos = np.zeros(Ndense)

    row_spacing = (Ndense - lastpoint_bool) / (Nsparse - lastpoint_bool)

    # round away from zero for n.5
    row_indices = \
        (np.trunc((row_spacing * np.arange(Nsparse)) + 0.5)).astype(int)

    # round towards even integers for n.5
    # row_indices = np.round((row_spacing*np.arange(Nsparse))

    # set half-wave to 1 at indicated places
    row_mask_pos[row_indices] = 1

    # make the length 2*Ndense output array, according to whether offbool is on
    # Note: I wonder if it would make more sense, for the offbool = 0 case, to
    #       set the first point = 1 instead of nan. That way you'd have the
    #       same np.nansum(row_mask_out) in both cases. In that case, the first
    #       point is always a zero-pad anyway, so we can probably say we
    #       "measured" it?
    if offbool:
        row_mask_out = np.concatenate((row_mask_pos[::-1], row_mask_pos))
    else:
        row_mask_out = np.concatenate(([np.nan], row_mask_pos[:0:-1],
                                       row_mask_pos))
        # row_mask_out = np.concatenate(([1], row_mask_pos[:0:-1],
        # row_mask_pos))

    return row_mask_out


def stagger_sample(N3d, Ndense, Nsparse, offbool):
    """
    This function takes the size of a 3d set of data (multiple 2d data sets),
    and an Nt1 value, and makes a sampling pattern that is staggered across the
    2d data sets in order to fill out every t1 value, where there are N1 bins
    along t1.

    Visualize with e.g. the following code:

    sampling_mask_2=stagger_sample(10,128,30,1)
    fig, ax = plt.subplots(figsize=(10,5))
    for i in np.arange(10):
        ax.bar(np.arange(len(sampling_mask_2[i])),sampling_mask_2[i],1.0,9-i,align='center')
    ax.set_xlim(0,256)
    ax.set_ylim(0,10)
    """

    # get a quasi-even 1d sampling pattern
    sampling_mask_1 = sampling_mask(Ndense, Nsparse, offbool, 1)
    sampling_mask_2 = sampling_mask_1.copy()
    # start off the sampling mask with this pattern
    stagger_sampling_mask = []

    # from here on down, Ndense refers to the full size of the sampling mask,
    # so e.g. 128 -> 256 Ndense*=2

    # now begin "stepping" the pattern further and further out, always keeping
    # the first point.
    for i in np.arange(N3d):

        sampling_mask_2[Ndense - 1:Ndense + 1] = 1
        stagger_sampling_mask.append(list(sampling_mask_2))
        sampling_mask_1 = sampling_mask_2.copy()

        sampling_mask_2[:] = np.full(2 * Ndense, np.nan)

        # in each case below, keep the middle one point (or middle two points)
        # unchanged; should always be 1
        if(offbool == 1):
            # left side: shift all to left, but then make sure the 0th points
            # wraps to the middle
            sampling_mask_2[0:Ndense - 1] = sampling_mask_1[1:Ndense]
            sampling_mask_2[Ndense - 2] = sampling_mask_1[0]
            # right side: shift all to right, but then make sure the Nth point
            # wraps to the middle
            sampling_mask_2[Ndense + 1:2 * Ndense] = \
                sampling_mask_1[Ndense:2 * Ndense - 1]
            sampling_mask_2[Ndense + 1] = sampling_mask_1[2 * Ndense - 1]
        else:
            # if offbool=0, we have an empty "NaN" at the 0 index -- so IGNORE
            # the 0 point! left side: shift all to left, but then make sure the
            # 0th points wraps to the middle
            sampling_mask_2[1:Ndense] = sampling_mask_1[2:Ndense + 1]
            sampling_mask_2[Ndense - 1] = sampling_mask_1[1]
            # right side: shift all to right, but then make sure the Nth point
            # wraps to the middle
            sampling_mask_2[Ndense + 1:2 * Ndense] = \
                sampling_mask_1[Ndense:2 * Ndense - 1]
            sampling_mask_2[Ndense + 1] = sampling_mask_1[2 * Ndense - 1]

    return stagger_sampling_mask


def sparsify_staggered(data, Nsparse, N3D, offbool):
    """
    "Undersamples" a dense data set using the staggered pattern.
    **
    *** THE INPUT DATA SET MUST INCLUDE THE DATA YOU WANT FROM *ALL* 2D SLICES
        *ALONG THE 3RD DIM. ***
    **
    This function figures out the sampling
    pattern for the given Nsparse and offbool (and len(data)), and sets all
    points that aren't in that sampling set to 0.

    IMPORTANT: Returns a tuple of arrays (sparse_data, measured_points).
               These are the same except the former has 0s and the latter has
               nans. They both are needed for run_codiffmap_1d()!

    Notes: data should be the "MRI-like" data, with both t < 0 and t > 0 parts.
           Nsparse must be <= 0.5*len(data).
    """
    # Get number of dense points & use that to create a row mask.
    # If Nsparse > Ndense, yell at the user and return the input
    # data without modification.

    # Note: the data as input has 'data[2d slice][f2][t1]', so index
    # accordingly!
    Ndense = np.int(0.5 * len(data[0][0]))
    sparse_data = np.copy(data)
    measured_points = np.copy(data)

    if Ndense >= Nsparse:
        stagger_sampling_mask = stagger_sample(N3D, Ndense, Nsparse, offbool)
        push_points = make_push_points(data, stagger_sampling_mask)

        # for this function, we need to repeat these steps for each of the data
        # sets along the 3rd dim. and also along all included columns
        # NOTICE: stagger_sampling_mask is only 2D, it DOES NOT have a 3rd
        # dimension! (sampling identical for all columns)
        for i in np.arange(N3D):
            for j in np.arange(len(data[0])):
                sparse_data[i][j][np.isnan(stagger_sampling_mask[i])] = 0
                measured_points[i][j] *= stagger_sampling_mask[i]
    else:
        print("ERROR: Nsparse > Ndense !!")
        print("Returning the input data unmodified.")

    return sparse_data, measured_points, push_points


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

    if len(data.shape) == 2:
        Nspectra = 1
    else:
        Nspectra = data.shape[0]

    # initialize the data shape and amp_list to output, and the phase
    # correction
    N1 = data.shape[-1]
    amp_list = peak_list.copy()
    pcorr_f1 = get_phasecorr(N1, 1)

    # we output a peak for each 2d spectrum: peak_amp[spectrum][peak]
    if Nspectra == 1:
        peak_gridamps = np.full((peak_list.shape[0]), np.nan)
    else:
        peak_gridamps = np.full((Nspectra, peak_list.shape[0]), np.nan)

    # fill out the grid of data: each spectrum, each peak
    for spectrum in np.arange(Nspectra):
        for peak_num in np.arange(peak_list.shape[0]):
            peak_row = int(peak_list[peak_num, 0])
            peak_col = int(peak_list[peak_num, 1])

            if Nspectra == 1:
                data_left_t = data[peak_row - 1, :]
                data_center_t = data[peak_row, :]
                data_right_t = data[peak_row + 1, :]
            else:
                data_left_t = data[spectrum][peak_row - 1, :]
                data_center_t = data[spectrum][peak_row, :]
                data_right_t = data[spectrum][peak_row + 1, :]

            data_left_f = np.fft.fftshift(np.fft.fft(data_left_t))
            data_center_f = np.fft.fftshift(np.fft.fft(data_center_t))
            data_right_f = np.fft.fftshift(np.fft.fft(data_right_t))

            # phase correct into a real spectrum
            data_left_fr = (data_left_f * pcorr_f1).real
            data_center_fr = (data_center_f * pcorr_f1).real
            data_right_fr = (data_right_f * pcorr_f1).real

            amp = 0
            amp += grid_bool * np.sum(data_left_fr[peak_col - 1:peak_col + 2])
            amp += np.sum(data_center_fr[peak_col - grid_bool * 1:
                                         peak_col + 1 + grid_bool])
            amp += grid_bool * np.sum(data_right_fr[peak_col - 1:peak_col + 2])

            if Nspectra == 1:
                peak_gridamps[peak_num] = amp
            else:
                peak_gridamps[spectrum][peak_num] = amp

    return peak_gridamps
