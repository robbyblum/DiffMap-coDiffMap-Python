# utility functions for diffmap and codiffmap
# getting them all in one place

import numpy as np


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


def get_phasecorr2D(dimensions, axes=None, offbool=(False, False)):
    """
    Creates a phase correction array, to make the FFT of a Hermitian
    array all real. This function, unlike get_phasecorr, outputs a 2D array!

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the output array. Should match whatever you want to apply
        this array to (i.e., use foo.shape if you want to apply this to an
        array foo)
    axes : int or shape tuple, optional
        explanation
    offbool : tuple, optional
        explanation

    Returns
    -------
    pcorrmat : ndarray
        phase correction array.

    """
    if axes is None:
        axes = (0, 1)
    elif isinstance(axes, np.compat.integer_types):
        axes = (axes,)

    N0, N1 = dimensions
    offbool0, offbool1 = offbool

    # define shift amounts, with optional offset by 1/2 for the indirect
    # dimension
    shift0 = (N0 / 2 - 0.5 * offbool0)
    shift1 = (N1 / 2 - 0.5 * offbool1)

    # the "neither 0 nor 1 in axes" case is trivial, but it might as well
    # be fast, so only make these if they're needed
    if (0 in axes) or (1 in axes):
        n0, n1 = np.indices(dimensions)

    pcorrmat = np.ones(dimensions, dtype='complex128')
    if 0 in axes:
        # integer division here; and half-step shift in the zero location
        # should be taken care of by offbool
        n0 -= N0 // 2
        pcorrmat *= np.exp(-1.j * 2 * np.pi * n0 * shift0 / N0)
    if 1 in axes:
        n1 -= N1 // 2
        pcorrmat *= np.exp(-1.j * 2 * np.pi * n1 * shift1 / N1)

    return pcorrmat


def states_to_mri(data, p0=0, offbool=(False, False), invert_sin=False):
    """
    converts states-like data to mri-like form. assumes that you have already
    done any zero-filling and apodization that you want to do, but allows you
    to apply a global zeroth order phase shift if you want.
    """
    # First, get the array dimensions we'll need for the calculation.
    # Because the initial array is hypercomplex in t1, the final number of
    # t1 points, N1_f, is the same as the initial number, N1_i. We have to
    # double the number of t2 points, though, so N2_f = 2*N2_i.
    N1_i, N2_i = data.shape
    N1_f = N1_i
    N2_f = 2 * N2_i
    N1mid = N1_f // 2
    N2mid = N2_f // 2

    offbool1, offbool2 = offbool

    # Apply a zeroth order phase shift, if desired
    data = ng.proc_base.ps(data, p0=p0)

    dataA = np.zeros((N1mid, N2_i))
    dataB = np.zeros((N1mid, N2_i))
    dataC = np.zeros((N1mid, N2_i))
    dataD = np.zeros((N1mid, N2_i))
    # If we think of the hypercomplex input data as two different arrays,
    # S_cos(t1, t2) and S_sin(t1, t2), interleaved along t1, then we can
    # break it down into components as follows:
    #     dataA = S_cos(|t1|, |t2|).real
    #     dataB = S_cos(|t1|, |t2|).imag
    #     dataC = S_sin(|t1|, |t2|).real
    #     dataD = S_sin(|t1|, |t2|).imag
    # These arrays are all of shape (N1mid, N2_i).
    dataA = (data[::2, :]).real
    dataB = (data[::2, :]).imag
    dataC = (-1)**(invert_sin) * (data[1::2, :]).real
    dataD = (-1)**(invert_sin) * (data[1::2, :]).imag

    data_out = np.zeros((N1_f, N2_f), dtype=np.complex)

    # Construct data_out one quadrant at a time. How this works depends
    # on offbool in a slightly complicated way, so build the slice
    # objects for each quadrant first. There's probably a fancier way
    # to do the logic here (esp. for q3), but if so, I don't know it.

    # QI (t1>=0, t2>=0) is always the same:
    q1 = np.s_[N1mid:, N2mid:]

    # QII (t1>=0, t2<=0) depends on offbool2:
    if offbool2:
        q2 = np.s_[N1mid:, N2mid - 1::-1]
    else:
        q2 = np.s_[N1mid:, N2mid:0:-1]

    # QIII (t1<=0, t2<=0) depends on offbool1 AND offbool2:
    if offbool1 and offbool2:
        q3 = np.s_[N1mid - 1::-1, N2mid - 1::-1]
    elif offbool1:
        q3 = np.s_[N1mid - 1::-1, N2mid:0:-1]
    elif offbool2:
        q3 = np.s_[N1mid:0:-1, N2mid - 1::-1]
    else:
        q3 = np.s_[N1mid:0:-1, N2mid:0:-1]

    # QIV (t1<=0, t2>=0) depends on offbool1:
    if offbool1:
        q4 = np.s_[N1mid - 1::-1, N2mid:]
    else:
        q4 = np.s_[N1mid:0:-1, N2mid:]

    data_out[q1] += dataA - dataD + 1j * (dataB + dataC)
    data_out[q2] += dataA + dataD + 1j * (-dataB + dataC)
    data_out[q3] += dataA - dataD + 1j * (-dataB - dataC)
    data_out[q4] += dataA + dataD + 1j * (dataB - dataC)

    # in the offboolN = False case, we end up double-counting the
    # tN=0 data, so we have to divide that row/column by 2 to fix it
    if not offbool1:
        data_out[N1mid, :] /= 2
    if not offbool2:
        data_out[:, N2mid] /= 2

    return data_out


def sampling_mask(Ndense, Nsparse, offbool, lastpoint_bool=0):
    """
    Makes a "row mask" array, for a given number of dense points and
    a given number of sparse points. Ndense is number of dense points
    in t >= 0! The output wave will be length (2*Ndense).

    This version of sampling_mask has a ensures we have sampled the ``last'',
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

    row_space = (Ndense - lastpoint_bool) / (Nsparse - lastpoint_bool)

    # round away from zero for n.5
    row_inds = (np.trunc((row_space * np.arange(Nsparse)) + 0.5)).astype(int)

    # round towards even integers for n.5
    # row_inds = np.round((row_spacing*np.arange(Nsparse)).astype(int)

    # set half-wave to 1 at indicated places
    row_mask_pos[row_inds] = 1

    # make the length 2*Ndense output array, according to whether offbool is on
    # Note: for offbool = 0, we set the first point = 1 instead of nan. This
    #       way, np.nansum(row_mask_out) is the same in both cases.
    if offbool:
        row_mask_out = np.concatenate((row_mask_pos[::-1], row_mask_pos))
    else:
        # row_mask_out = np.concatenate(([np.nan], row_mask_pos[:0:-1],
        #                                row_mask_pos))
        row_mask_out = np.concatenate(([1], row_mask_pos[:0:-1], row_mask_pos))

    return row_mask_out
