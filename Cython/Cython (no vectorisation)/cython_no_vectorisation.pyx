# .pyx file where I will define the optimized (cythonised) functions of the original Lebwohl-Lasher program: 
# initdat, one_energy, all_energy, get_order, MC_step

"""
CYTHON OPTIMISATIONS MADE:

1. Disabling bounds checking to access arrays faster using cython.boundscheck(False)
2. Disabling negative index wraparound to reduce overhead using cython.wraparound(False)
3. Static c-style typing for all variables, function inputs and function outputs(explicit defining of varibale types like int, double etc..)
4. using cmath for mathematical operations instead of numpy (e.g. cmath.cos, cmath.exp)

"""


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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double all_energy(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
    
    cdef double enall = 0.0
    cdef int i, j

    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)

    return enall

#=======================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double get_order(np.ndarray[np.float64_t, ndim=2] arr, int nmax):
   
    cdef np.ndarray[np.float64_t, ndim=2] Qab = np.zeros((3,3))
    cdef np.ndarray[np.float64_t, ndim=2] delta = np.eye(3,3)
    cdef int a, b, i, j
    
    cdef np.ndarray[np.float64_t, ndim=3] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)

    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]

    Qab = Qab/(2*nmax*nmax)

    eigenvalues,eigenvectors = np.linalg.eig(Qab)

    return eigenvalues.max()
#=======================================================================


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double MC_step(np.ndarray[np.float64_t, ndim=2] arr, double Ts, int nmax):

    cdef double scale=0.1+Ts
    cdef int accept = 0
    cdef int i, j, ix, iy
    cdef double ang, en0, en1, boltz
    cdef np.ndarray[np.int_t, ndim=2] xran, yran
    cdef np.ndarray[np.float64_t, ndim=2] aran

    xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    aran = np.random.normal(scale=scale, size=(nmax,nmax))

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]

            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)

            if en1<=en0:
                accept += 1

            else:
                boltz = exp( -(en1 - en0) / Ts )      # use cmath.exp instead of np.exp

                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang

    return accept/(nmax*nmax)