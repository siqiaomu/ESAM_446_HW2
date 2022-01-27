
#NEED TO MAKE AXIS WORK

import numpy as np
import scipy.fft

class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

    def transform_to_grid(self, data, axis, dtype, scale=1):
        if dtype == np.complex128:
            return self._transform_to_grid_complex(data, axis, scale)
        elif dtype == np.float64:
            return self._transform_to_grid_real(data, axis, scale)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def transform_to_coeff(self, data, axis, dtype):
        if dtype == np.complex128:
            return self._transform_to_coeff_complex(data, axis)
        elif dtype == np.float64:
            return self._transform_to_coeff_real(data, axis)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def _transform_to_grid_complex(self, data, axis, scale):
        if scale == 1:
            grid_data = (scale * self.N) * scipy.fft.ifft(data)
        elif scale > 1:
            coeff_data = np.zeros(int(self.N * scale), dtype=np.complex128)
            first_half_data = data[0:int(self.N/2)]
            second_half_data = data[int(self.N/2 + 1):]
            coeff_data[0:int(self.N/2)] = first_half_data
            coeff_data[-int(self.N/2 - 1):] = second_half_data
            grid_data = (scale * self.N) * scipy.fft.ifft(coeff_data)
        return grid_data

    def _transform_to_coeff_complex(self, data, axis):
            
        coeff_data =  scipy.fft.fft(data)
        if len(data) == self.N:
            return (1/self.N) * coeff_data
        else:
            first_half_data = coeff_data[0:int(self.N/2)]
            second_half_data = coeff_data[-int(self.N/2):]
            return (1/len(data)) * np.append(first_half_data, second_half_data)
        

    def _transform_to_grid_real(self, data, axis, scale):
        complex_data = np.zeros(self.N//2+1, dtype=np.complex128)
        complex_data[:int(self.N/2)].real = data[::2]
        complex_data[:int(self.N/2)].imag = data[1::2]
        
        if scale == 1:
            grid_data = scipy.fft.irfft(complex_data)
        elif scale > 1:
            expanded_complex_data = np.zeros(int(scale*self.N)//2+1, dtype=np.complex128)
            expanded_complex_data[:int(self.N/2)] = complex_data[:int(self.N/2)]
            grid_data = scipy.fft.irfft(expanded_complex_data)
        return (self.N * scale/2) *grid_data
            
            
    def _transform_to_coeff_real(self, data, axis):
        coeffs = scipy.fft.rfft(data)
        coeff_data = np.zeros(self.N)
        coeff_data[::2] = coeffs.real[:int(self.N/2)]
        coeff_data[1::2] = coeffs.imag[:int(self.N/2)]
        return (2/len(data)) * coeff_data

class Domain:

    def __init__(self, bases):
        if isinstance(bases, Basis):
            # passed single basis
            self.bases = (bases, )
        else:
            self.bases = tuple(bases)
        self.dim = len(self.bases)

    @property
    def coeff_shape(self):
        return [basis.N for basis in self.bases]

    def remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if not hasattr(scales, "__len__"):
            scales = [scales] * self.dim
        return scales


class Field:

    def __init__(self, domain, dtype=np.float64):
        self.domain = domain
        self.dtype = dtype
        self.data = np.zeros(domain.coeff_shape[0], dtype=dtype)
        self.coeff = np.array([True]*self.data.ndim)

    def towards_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        axis = np.where(self.coeff == False)[0][0]
        self.data = self.domain.bases[axis].transform_to_coeff(self.data, axis, self.dtype)
        self.coeff[axis] = True

    def require_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        else:
            self.towards_coeff_space()
            self.require_coeff_space()

    def towards_grid_space(self, scales=None):
        if not self.coeff.any():
            # already in full grid space
            return
        axis = np.where(self.coeff == True)[0][-1]
        scales = self.domain.remedy_scales(scales)
        self.data = self.domain.bases[axis].transform_to_grid(self.data, axis, self.dtype, scale=scales[axis])
        self.coeff[axis] = False

    def require_grid_space(self, scales=None):
        if not self.coeff.any(): 
            # already in full grid space
            return
        else:
            self.towards_grid_space(scales)
            self.require_grid_space(scales)



