# .pyx file where I will define the optimized (cythonised) functions of the Lebwohl-Lasher program: 
# initdat, one_energy, all_energy, get_order, MC_step
# Note that the 'get_order' and 'all_energy' functions are vectorised (from the V2 NumPy vectorisation script)
# The random numbers are precomputed for the entire lattice (outside the loop) for the Boltzmann Comparison in MC_step

"""
CYTHON OPTIMISATIONS MADE:

1. Disabling bounds checking to access arrays faster using cython.boundscheck(False)
2. Disabling negative index wraparound to reduce overhead using cython.wraparound(False)
3. Static c-style typing for all variables (explicit defining of varibale types like int, double etc..)
4. using cmath for mathematical operations instead of numpy (e.g. cmath.cos, cmath.exp). Note this only works on scalars, so numpy was used for arrays

"""
# MPI has also been applied for parallelisation

# Import libraries
import cython
cimport cython 
import numpy as np 
cimport numpy as np 
from libc.math cimport cos, exp 

np.import_array() # Initialise NumPy array interface so they can be used 
 
#=======================================================================
@cython.boundscheck(False)  
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] initdat(int nmax) noexcept:

    return np.random.random_sample((nmax,nmax))*2.0*np.pi
#=======================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double one_energy(np.ndarray[np.float64_t, ndim=2] arr, int ix, int iy, int nmax):

    cdef double en = 0.0
    cdef int ixp = (ix + 1) % nmax
    cdef int ixm = (ix - 1) % nmax
    cdef int iyp = (iy + 1) % nmax
    cdef int iym = (iy - 1) % nmax
    cdef double ang

    # Replace np.cos with cmath.cos
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0* cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0* cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0* cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0* cos(ang)**2)

    return en
#=======================================================================

# Vectorised all_energy function 
@cython.boundscheck(False)  
@cython.wraparound(False)
cpdef double all_energy(np.ndarray[np.float64_t, ndim=2] arr, int nmax):

    cdef np.ndarray[np.float64_t, ndim=2] shift_left = np.roll(arr, 1, axis=1)
    cdef np.ndarray[np.float64_t, ndim=2] shift_right = np.roll(arr, -1, axis=1)
    cdef np.ndarray[np.float64_t, ndim=2] shift_up = np.roll(arr, -1, axis=0)
    cdef np.ndarray[np.float64_t, ndim=2] shift_down = np.roll(arr, 1, axis=0)
    
    cdef np.ndarray[np.float64_t, ndim=2] energy_l = 0.5 * (1.0 - 3.0 * np.cos(arr - shift_left) ** 2)
    cdef np.ndarray[np.float64_t, ndim=2] energy_r = 0.5 * (1.0 - 3.0 * np.cos(arr - shift_right) ** 2)
    cdef np.ndarray[np.float64_t, ndim=2] energy_up = 0.5 * (1.0 - 3.0 * np.cos(arr - shift_up) ** 2)
    cdef np.ndarray[np.float64_t, ndim=2] energy_down = 0.5 * (1.0 - 3.0 * np.cos(arr - shift_down) ** 2)

    cdef double enall = np.sum(energy_l + energy_r + energy_up + energy_down)

    return enall

#=======================================================================

# Vectorised get_order function 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double get_order(np.ndarray[np.float64_t, ndim=2] arr, int nmax):

    cdef np.ndarray[np.float64_t, ndim=3] lab = np.stack((np.cos(arr), np.sin(arr), np.zeros_like(arr)), axis=0)
    cdef np.ndarray[np.float64_t, ndim=2] delta = np.eye(3)
    cdef np.ndarray[np.float64_t, ndim=2] Qab = np.tensordot(lab, lab, axes=([1, 2], [1, 2])) * 3 - delta * (nmax * nmax)

    Qab = Qab / (2 * nmax * nmax)
    cdef np.ndarray[np.float64_t, ndim=1] eigenvalues = np.linalg.eigvalsh(Qab)

    return np.max(eigenvalues)

#=======================================================================

# Updated MC_step function 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double MC_step(np.ndarray[np.float64_t, ndim=2] arr, double Ts, int rankID, int n_proc, int nmax):

    # Determine the section if the lattice for each worker (splitting by rows)
    rows_per_process = nmax // n_proc
    row_start = rankID * rows_per_process # the starting row for current worker 
    row_end = (rankID + 1) * rows_per_process if rankID != n_proc-1 else nmax  # last row for current worker 

    cdef double scale=0.1+Ts
    cdef int accept = 0
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz
    cdef np.ndarray[np.int_t, ndim=2] xran, yran
    cdef np.ndarray[np.float64_t, ndim=2] aran, boltz_random

    xran = np.random.randint(0,high=nmax, size=(row_end-row_start, nmax))   # edit for mpi
    yran = np.random.randint(0,high=nmax, size=(row_end-row_start,nmax))
    aran = np.random.normal(scale=scale, size=(row_end-row_start,nmax))

    boltz_random = np.random.uniform(0.0, 1.0, size=(nmax, nmax))

    for i in range(row_start, row_end):
        for j in range(nmax):
            ix = xran[i - row_start,j]
            iy = yran[i - row_start,j]
            ang = aran[i - row_start,j]

            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)

            # Boltzmann factor (criterion)
            boltz = exp( -(en1 - en0) / Ts )

          
            if en1<=en0 or boltz >= boltz_random[i,j]:
                accept += 1   # accept the new configuration 
            else:
                arr[ix,iy] -= ang   # Reject the new configuration (revert angle)
            
    return accept

    #=======================================================================
