# Python implementation of DiffMap code
# The first step is to move the functions from a Jupyter notebook over to here
# functions are in no particular order yet
# TODO: make sure it's nicely interoperable with nmrglue
import numpy as np


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


def lin_alg_limit(mask_1D):
    """
    Calculate the "linear algebra limit" of a given 1D mask array.

    This is just the number of non-zero elements in the array, divided by 2.
    Takes an nparray, returns a number (the linear algebra limit).
    It's dirt simple, but useful to have in this form nonetheless.
    """

    # lin_alg_lim = 0.5 * np.count_nonzero(mask_1D)

    return 0.5 * np.count_nonzero(mask_1D)


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


def run_difference_map(data_in, N_iter, measured_points, mask, offbool,
                       conv_t=False, conv_f=False, conv_delta=5e-6):
    """
    Runs the difference map on data_in, with N_iter iterations.
    Outputs P2(D(data_in)^N); that is; it applies P2 at the end of the
    iteration loop.
    Also has the option to calculate L1 and do convergence testing based on
    (relative) change in L1.

    TODO: maybe change the inputs so that measured_points is generated from
          data_in? Might need to input a row mask or an Nsparse value instead.
    """
    data_out = data_in.copy()

    if conv_t:
        L1_t = np.zeros(N_iter + 1)
        L1_t[0] = np.linalg.norm(data_in, 1)
        L1_t_diff = np.zeros(6)

    if conv_f:
        L1_f = np.zeros(N_iter + 1)
        L1_f[0] = np.linalg.norm(np.fft.fft(data_in), 1)
        L1_f_diff = np.zeros(6)

    # Infer offbool from the form of the given data...may not be stable?
    # Update: It's not stable, alas. I would want to choose some sort of
    # equality tolerance to do this, I think. Not a priority right now.
    # N = len(data_in)
    # offbool = data_in.real[int(N/2)] == data_in.real[int(N/2 - 1)]
    # offbool = 1

    # do the difference map N_iter times
    for n in np.arange(N_iter):
        data_out = difference_map(data_out, measured_points, mask, offbool)

        # do the L1 calculations. You don't need to do the fftshift or phase
        # correction, because they don't affect the outcome of L1
        if conv_t:
            p2_out = p2_proj(data_out, measured_points)
            L1_t[n + 1] = np.linalg.norm(p2_out, 1)

            # if for some reason you're doing both convergence tests,
            # don't do P2 twice
            if conv_f:
                p2_out_fft = np.fft.fft(p2_out)
                L1_f[n + 1] = np.linalg.norm(p2_out_fft, 1)

        elif conv_f:
            p2_out_fft = np.fft.fft(p2_proj(data_out, measured_points))
            L1_f[n + 1] = np.linalg.norm(p2_out_fft, 1)

        if conv_t or conv_f and n > 5:
            if conv_t:
                L1_t_diff = np.abs(L1_t[n + 1] - L1_t[n - np.arange(6)]) \
                    / L1_t[n + 1]

                if np.max(L1_t_diff) < conv_delta:
                    break

            if conv_f:
                L1_f_diff = np.abs(L1_f[n + 1] - L1_f[n - np.arange(6)]) \
                    / L1_f[n + 1]

                if np.max(L1_f_diff) < conv_delta:
                    break

    # apply final P2 projection, to restore the measured data points
    data_out = p2_proj(data_out, measured_points)

    # if using convergence test(s), return L1 and number of iterations as well
    if conv_t and conv_f:
        return data_out, L1_t, L1_f, n + 1
    elif conv_t:
        return data_out, L1_t, n + 1
    elif conv_f:
        return data_out, L1_f, n + 1
    else:
        return data_out


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


def p2_proj(data_t, measured_points):
    """
    Applies the P2 projection to a 1D array 'data_t'.

    Overwrites some of the points in 'data_t' with original measured values,
    which are stored in 'measured_points'. 'measured_points' is NaN for points
    that weren't sampled, and equal to the measured values for points that
    were sampled.
    """
    # copy data to output array, force to be complex
    p2_data_t = data_t.astype(complex)

    p2_data_t[~np.isnan(measured_points)] = \
        measured_points[~np.isnan(measured_points)]

    return p2_data_t


def difference_map(data_t, measured_points, mask, offbool):
    """
    Applies one step of the difference map algorithm to a 1D array 'data_t'.

    Uses the functions p1_proj and p2_proj internally.
    D = 1 + P1 (2 P2 - 1) - P2
    """

    # calculate (2 P2 - 1)[data_t] first, for clarity's sake
    data_temp = 2 * p2_proj(data_t, measured_points) - data_t

    d_data_t = data_t + p1_proj(data_temp, mask, offbool) \
        - p2_proj(data_t, measured_points)

    return d_data_t


def sampling_mask(Ndense, Nsparse, offbool):
    """
    Makes a "row mask" array, for a given number of dense points and
    a given number of sparse points. Ndense is number of dense points
    in t >= 0! The output wave will be length (2*Ndense).

    if offbool == True, then the central point (at t = dw/2) is reflected
    to the t < 0 side of the vector. If offbool == False, then the central
    point is at t = 0 and isn't duplicated.

    NOTE: the rounding convention in Igor is "away from zero." In numpy/py3
    it's "towards even." I'm implementing the Igor version here, but I might
    change it to the python version later. It _will_ change the row choices in
    some cases, though!

    TODO: enforce 1 <= Nsparse <= Ndense properly
    """
    # initialize positive side as nans, not zeroes
    row_mask_pos = np.full(Ndense, np.nan)
    # row_mask_pos = np.zeros(Ndense)

    row_space = Ndense / Nsparse

    # round away from zero for n.5
    row_inds = (np.trunc((row_space * np.arange(Nsparse)) + 0.5)).astype(int)

    # round towards even integers for n.5
    # row_inds = np.round((row_space*np.arange(Nsparse))

    # set half-wave to 1 at indicated places
    row_mask_pos[row_inds] = 1

    # make the length 2*Ndense output array, according to whether offbool is on
    # Note: I wonder if it would make more sense, for the offbool = 0 case, to
    #       set the first point = 1 instead of nan. That way you'd have the
    #       same np.nansum(row_mask_out) in both cases. In that case, the first
    #       point is always a zero-pad anyway, so we can probably say we
    #       "measured" it?
    # Update, much later: Yes, that would be a good idea!
    if offbool:
        row_mask_out = np.concatenate((row_mask_pos[::-1], row_mask_pos))
    else:
        # row_mask_out = np.concatenate(
        #     ([np.nan], row_mask_pos[:0:-1], row_mask_pos))
        row_mask_out = np.concatenate(([1], row_mask_pos[:0:-1], row_mask_pos))

    return row_mask_out


def sparsify(data, Nsparse, offbool):
    """
    "Undersamples" a dense data set. This function figures out the sampling
    pattern for the given Nsparse and offbool (and len(data)), and sets all
    points that aren't in that sampling set to 0.

    IMPORTANT: Returns a tuple of arrays (sparse_data, measured_points).
               These are the same except the former has 0s and the latter has
               nans. They both are needed for run_difference_map()!

    Notes: data should be the "MRI-like" data, with both t < 0 and t > 0 parts.
           Nsparse must be <= 0.5*len(data).
    """
    # Get number of dense points & use that to create a row mask.
    # If Nsparse > Ndense, yell at the user and return the input
    # data without modification.
    Ndense = len(data) // 2
    sparse_data = data.copy()
    measured_points = data.copy()

    if Ndense >= Nsparse:
        row_mask = sampling_mask(Ndense, Nsparse, offbool)

        sparse_data[np.isnan(row_mask)] = 0
        measured_points *= row_mask
    else:
        print("ERROR: Nsparse > Ndense !!")
        print("Returning the input data unmodified.")

    return sparse_data, measured_points


def alias_overlap_wrap(mask, peaki, Nt1max):
    """
    Calculates the alias overlap metric (wraparound version) for a given peak
    location.

    Returns an array of length Nt1max giving the metric results for each Nt1.
    At each Nt1, alias_out has the cumulative number of "mask hits" across all
    the alias harmonics checked.
    """
    # initialize output wave. Set index 0 to nan, because it's not meaningful
    alias_out = np.zeros(Nt1max)
#     alias_out[0] = np.nan
    alias_out[0] = 128

    # length of the mask vector
    N = len(mask)

    # array of indices of nonzero mask elements
    masknz = mask.nonzero()[0]

    # k = number of harmonics to calculate
    # I haven't proven this, but I believe you don't need more than Nt1max/2.
    # Going up to Nt1max just to be safe. Anything beyond that just adds
    # numbers to alias_out[1], forever.
    for k in range(1, Nt1max + 1):

        # loop over valid Nt1 values for harmonic k: Nt1 <= np.ceil(Nt1max/k)
        for Nt1 in range(1, int(np.ceil(Nt1max / k) + 1)):

            # Locate our points of interest, where we'll look for mask points.
            # As Nt1+=1, jump usually changes by 2, so look at the "next
            #     closer" point as well
            # Python modulus (a % b) takes the sign of b,
            #     so this is simpler than in Igor!
            jump = Nt1 * k * N / Nt1max
            i_alias = np.mod((peaki + jump, peaki + jump - 1,
                              peaki - jump, peaki - jump + 1), N)

            # self-collisions don't count
            if peaki not in i_alias:
                # add up instances of i_alias in masknz
                alias_out[Nt1] += sum(np.in1d(i_alias, masknz))

    return alias_out


def alias_overlap_wrap3(mask, peaki, Nt1max, beta):
    """
    Calculates the alias overlap metric (wraparound version) for a given peak
    location.

    Returns an array of length Nt1max giving the metric results for each Nt1.
    At each Nt1, alias_out has the cumulative number of "mask hits" across all
    the alias harmonics checked.

    wrap3 checks 3 points instead of 2 for each alias location!
    beta is a "harmonic cutoff parameter" that can be between 0 and 1, and
    controls how many Nt1 are calculated for a given k.
    """
    # initialize output wave. Set index 0 to nan, because it's not meaningful
    alias_out = np.zeros(Nt1max)
#     alias_out[0] = np.nan
    alias_out[0] = 128

    # length of the mask vector
    N = len(mask)

    # array of indices of nonzero mask elements
    masknz = mask.nonzero()[0]

    # k = number of harmonics to calculate
    # I haven't proven this, but I believe you don't need more than Nt1max/2.
    # Going up to Nt1max just to be safe. Anything beyond that just adds
    # numbers to alias_out[1], forever.
    for k in range(1, Nt1max + 1):

        Nt1max_loop = np.floor(beta * Nt1max / k)

        # loop over valid Nt1 values for harmonic k: Nt1 <= np.ceil(Nt1max/k)
        for Nt1 in range(1, int(Nt1max_loop) + 1):

            # Locate our points of interest, where we'll look for mask points
            # As Nt1+=1, jump usually changes by 2, so look at the "next
            #     closer" and "next farther" points as well
            # Python modulus (a % b) takes the sign of b, so this is simpler
            #    than in igor
            jump = Nt1 * k * N / Nt1max
            i_alias = np.mod((peaki + jump + 1, peaki + jump, peaki + jump - 1,
                             peaki - jump - 1, peaki - jump, peaki - jump + 1),
                             N)

            # self-collisions don't count
            if peaki not in i_alias:
                # add up instances of i_alias in masknz
                alias_out[Nt1] += sum(np.in1d(i_alias, masknz))

    return alias_out


def mask_collision_metric(mask, peaki, Nt1max, beta):
    """
    wrapper for the alias overlap functions, probably not actually needed?
    all it does is calculate whether the values are nonzero.
    """

    return alias_overlap_wrap3(mask, peaki, Nt1max, beta) > 0


def setup_and_run_diffmap(data_in, Nsparse, mask, offbool, N_iter):
    """
    Shortcut to set up the required input data and run the diffmap function...
    This function is currently bad and non-general. It uses a hard-coded number
    of iterations, and doesn't have any options for convergence testing.
    """
    Ndense = np.int(len(data_in) / 2)

    # infer offbool from the form of the given data...may not be stable?
    # yeah, this isn't stable, alas. I could add in a fudge factor to the
    # equality, to account for machine precision, but ehhhhhh
    # offbool = data_in.real[Ndense] == data_in.real[Ndense-1]
    # print(offbool)

    data_gaps, meas_points = sparsify(data_in, Nsparse, offbool)

    data_out = run_difference_map(data_gaps, N_iter, meas_points, mask,
                                  offbool)

    return data_out


# def run_difference_map(data_in, N_iter, measured_points, mask, offbool):
#     """
#     Runs the difference map on data_in, with N_iter iterations.
#     Outputs P2(D(data_in)^N); that is; it applies P2 at the end of the
#     iteration loop.
#
#     TODO: maybe change the inputs so that measured_points is generated
#           from data_in? Might need to input a row mask or an Nsparse value
#           instead.
#
#     """
#     data_out = data_in.copy()
#
#     # infer offbool from the form of the given data...may not be stable?
#     # update: it's not stable, alas
#     # N = len(data_in)
#     # offbool = data_in.real[int(N/2)] == data_in.real[int(N/2 - 1)]
#     # offbool = 1
#     # do the difference map N_iter times
#     for n in np.arange(N_iter):
#         data_out = difference_map(data_out, measured_points, mask, offbool)
#
#     # apply final P2 projection, to restore the measured data points
#     data_out = p2_proj(data_out, measured_points)
#
#     return data_out
