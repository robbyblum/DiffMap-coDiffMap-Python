# -- coDiffMap
# -- May 13, 2019
# -- Core Python DiffMap functionality written by ROBERT BLUM
# -- coDiffMap adaptation written by JARED ROVNY

# -- processing
# 1. states_to_mri --> util.py
# 2. fft_phase_1d --> util.py
# 3. get_phasecorr --> util.py

# -- preparation
# 4. get_interesting_columns
# 5. get_interesting_mask_columns
# 6. get_interesting_columns_all2D

# -- sampling
# 7. sampling_mask --> util.py
# 8. stagger_sample
# 9. sparsify_staggered

# -- coDiffMap core
# 10. setup_and_run_codiffmap
# 11. run_codiffmap_1D
# 12. make_push_points
# 13. p1_proj
# 14. p2_proj_star
# 15. difference_map_star

import numpy as np
from . import util


# ---- data processing: all moved to util.py


# ---- data preparation

def get_interesting_columns(datapath, column_indices, spectrum_index, n_rows):
    """
    Load "interesting" columns of data from a given 2d data set.

    This function is designed to keep only the relevant columns in memory,
    since we need a lot of local data for the projection method.
    """

    data_source = str(datapath) + "_" + str(spectrum_index) + ".txt"
    # data_source = datapath

    real_cols = list(range(0, n_rows * 2, 2))
    imag_cols = list(range(1, n_rows * 2, 2))
    origdata_2d = np.loadtxt(data_source, usecols=real_cols) \
        - 1.j * np.loadtxt(data_source, usecols=imag_cols)

    origdata_2d_fft_ph = util.fft_phase_1d(origdata_2d)

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
    will just be from the columns chosen by the user. (actually it doesn't
    return a list anymore...)
    """

    n_columns = len(column_indices)

    all_relevant_data = np.zeros((number_of_spectra, n_columns, n_rows),
                                 dtype=np.complex)
    for spectrum_index in range(1, number_of_spectra + 1):
        new_data = get_interesting_columns(datapath, column_indices,
                                           spectrum_index, n_rows)
        all_relevant_data[spectrum_index - 1, :, :] = new_data
    return all_relevant_data


# ---- coDiffMap functions


def setup_and_run_codiffmap(data_in, Nsparse, mask, offbool, push_param,
                            N_iter):
    """
    Shortcut to set up the required input data and run the diffmap function...
    Note: mask needs to correspond to "data_in," column by column.
    """
    N3D, Ndense, Ncols = data_in.shape
    Ndense //= 2

    data_gaps, meas_points, push_points = sparsify_staggered(data_in, Nsparse,
                                                             offbool)
    data_out = np.copy(data_gaps)

    # temporarily hard-coding the axis here. It's probably still backwards.
    linalg_lims = np.abs(mask).sum(axis=0)

    for slice2d in np.arange(N3D):
        for column in np.arange(Ncols):
            # if there are no nonzero mask points in this column, skip it
            if linalg_lims[column] == 0:
                continue

            data_gaps_1d = data_gaps[slice2d, :, column]
            mask_1d = mask[:, column]
            meas_points_1d = meas_points[slice2d, :, column]
            push_points_1d = push_points[slice2d, :, column]

            data_out_1d = run_codiffmap_1D(data_gaps_1d, N_iter,
                                           meas_points_1d, mask_1d, offbool,
                                           push_points_1d, push_param)
            data_out[slice2d, :, column] = data_out_1d

    return data_out


def run_codiffmap_1D(data_in, N_iter, measured_points, mask, offbool,
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
    # offbool = 1
    # do the difference map N_iter times
    for n in np.arange(N_iter):
        data_out = difference_map_star(data_out, measured_points, mask,
                                       offbool, push_points, push_param)

    # apply final P2* projection, to restore the measured data points Note: for
    # the difference_map_star, it is important that this last step is a P2*,
    # not P2!
    data_out = p2_proj_star(data_out, measured_points, push_points, push_param)

    return data_out


def make_proxy_map(stagger_sampling_mask, offbool):
    """
    I'm splitting make_push_points into multiple functions. This one makes a
    proxy map: the output tells you which slice your proxy should come from,
    for every unsampled point.
    """
    nslice, nrow = stagger_sampling_mask.shape
    proxy_map = np.zeros((nslice, nrow))
    Ndense = nrow // 2

    for m in np.arange(nslice):
        proxy_map[m, ~np.isnan(stagger_sampling_mask[m, :])] = -1

        for n in np.arange(Ndense):
            if proxy_map[m, n] == 0:
                for spec2 in np.arange(2 * nslice):
                    # add +1, -1, +2, -2, +3, -3... to the spectrum index
                    dspec = m + (spec2 // 2 + 1) * (-1)**spec2
                    # make sure our new spectrum index isn't out of bounds
                    if 0 <= dspec < nslice:
                        # if the point has been sampled in this spectrum,
                        # use it, and stop looking
                        if stagger_sampling_mask[dspec, n] == 1:
                            proxy_map[m, n] = dspec
                            break
                # if there were no sampled point found, break with error.
                else:
                    print('ERROR: no suitable data point found!')
                    break
    if offbool:
        proxy_map[:, Ndense + 1:] = proxy_map[:, Ndense - 2::-1]
    else:
        proxy_map[:, Ndense + 1:] = proxy_map[:, Ndense - 1:0:-1]
    return proxy_map.astype(int)


def make_push_points(data, proxy_map):
    """
    New version of make_push_points. Uses the output of make_proxy_map.
    This function will take your data (many 2D slices), and a staggered
    sampling schedule, and make a data array of "push" points, f2-column by
    f2-column. For a given 2D slice, these push points will be:
    -- at measured t1 points: =NaN
    -- at unmeasured t1 points: is equal to the data at the
    same t1, but from a different (the closest available) 2D slice.
    """

    n_slices, n_rows, n_columns = data.shape

    push_points = np.zeros((n_slices, n_rows, n_columns), dtype=np.complex)

    bcast = proxy_map >= 0
    push_points[~bcast, :] = np.nan

    for n in np.arange(n_slices):
        pmap = proxy_map[n, bcast[n, :]]
        push_points[n, bcast[n, :], :] = data[pmap, bcast[n, :], :]

    return push_points


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
    phasecorr = util.get_phasecorr(len(data_f), offbool)

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


def stagger_sample(N3D, Ndense, Nsparse, offbool):
    """
    This function takes the size of a 3d set of data (multiple 2d data sets),
    and an Nt1 value, and makes a sampling pattern that is staggered across the
    2d data sets in order to fill out every t1 value, where there are N1 bins
    along t1.

    Visualize with e.g. the following code:

    sampling_mask_2=stagger_sample(10,128,30,1)
    fig, ax = plt.subplots(figsize=(10,5))
    for i in np.arange(10):
        ax.bar(np.arange(len(sampling_mask_2[i])), sampling_mask_2[i],
               1.0, 9-i, align='center')
    ax.set_xlim(0,256)
    ax.set_ylim(0,10)
    """

    # get a quasi-even 1d sampling pattern
    sample_mask = util.sampling_mask(Ndense, Nsparse, offbool, 1)
    # start off the sampling mask with this pattern
    stagger_sampling_mask = np.zeros((N3D, 2 * Ndense))
    stagger_sampling_mask[0, :] = sample_mask

    # now begin "stepping" the pattern further and further out, always keeping
    # the first point.
    for i in np.arange(1, N3D):
        # in each case below, keep the middle one point (or middle two points)
        # unchanged; should always be 1
        if(offbool == 1):
            # left side: roll all to left, making sure the 0th point wraps to
            # the middle
            sample_mask[0:Ndense - 1] = np.roll(sample_mask[0:Ndense - 1], -1)
            # right side: roll all to right, making sure the Nth point wraps to
            # the middle
            sample_mask[Ndense + 1:] = np.roll(sample_mask[Ndense + 1:], +1)
        else:
            # if offbool=0, we need to keep the 0 index equal to "1": so IGNORE
            # the 0th point! left side: roll all to left, making sure the 1st
            # point wraps to the middle
            sample_mask[1:Ndense] = np.roll(sample_mask[1:Ndense], -1)
            # right side: roll all to right, making sure the Nth point wraps to
            # the middle
            sample_mask[Ndense + 1:] = np.roll(sample_mask[Ndense + 1:], +1)

        stagger_sampling_mask[i, :] = sample_mask
    return stagger_sampling_mask


def sparsify_staggered(data, Nsparse, offbool):
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

    # Note: the data as input has 'data[2d_slice, f2, t1]', so index
    # accordingly!
    N3D, Ndense, Ncols = data.shape
    Ndense //= 2
    sparse_data = np.copy(data)
    measured_points = np.copy(data)

    if Ndense >= Nsparse:
        staggered_mask = stagger_sample(N3D, Ndense, Nsparse, offbool)
        proxy_map = make_proxy_map(staggered_mask, offbool)
        push_points = make_push_points(data, proxy_map)

        # for this function, we need to repeat these steps for each of the data
        # sets along the 3rd dim. and also along all included columns
        # NOTICE: stagger_sampling_mask is only 2D, it DOES NOT have a 3rd
        # dimension! (sampling identical for all columns)
        for i in np.arange(N3D):
            sparse_data[i, np.isnan(staggered_mask[i, :]), :] = 0
            measured_points[i, :, :] *= staggered_mask[i, :].reshape((-1, 1))
    else:
        print("ERROR: Nsparse > Ndense !!")
        print("Returning the input data unmodified.")

    return sparse_data, measured_points, push_points
