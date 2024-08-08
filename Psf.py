"""
    A collection of classes and functions to process PSFs including
    windowing according to location of local maxima
"""

import jax.numpy as np
from skimage.io import imread
import scipy.ndimage as filters
import matplotlib.pyplot as plt


class PSF:
    """
        Class PSF knows a 2D intensity array describing the PSF. 
        Class PSF can window the PSF according to the location of local maxima,
        and take the 2D FFT of the PSF.
    """
    def __init__(self,
                 image_array: np.array,
                 ) -> None:
        """
            Parameters:
            -----------
            image_array: np.array
                2D array of the PSF/intensity image
        """
        assert image_array.ndim == 2, "Input image array must be 2D"

        # Priv vars
        self._im_arr = image_array
        self._psf_dim = image_array.shape
        self._psf_center_coord = np.array([self._psf_dim[0]/2, self._psf_dim[1]/2], dtype=int)

        # For windowing. To set use method set_windowing_parameters. 
        # Defaulting to None for constructor clarity
        self._sub_psf_window_sz = None
        self._sigma = None 
        self._neighborhood_size = None 
        self._thresh_diff = None
        self._prox_thresh = None 
        self._parameters_set = False # parameter set flag

    # Alternative constructor
    @classmethod
    def from_file(cls, fname: str):
        """
            Parameters:
            -----------
            fname: str
                Path to the image file
            Returns:
            --------
            PSF object
        """
        im_array = imread(fname)

        return cls(image_array=im_array)
    
    def scale_intensity(self, upper_limit: float, lower_limit: float):
        """
            Parameters:
            -----------
            upper_limit: float
                Maximum intensity value to scale the image to
            lower_limit: float
                Minimum intensity value to scale the image to
                
            Returns:
            --------
            np.array
                Intensity-scaled image array, with values in range [lower_limit, upper_limit]
        """
        current_range = np.max(self._im_arr) - np.min(self._im_arr)
        new_range = upper_limit - lower_limit
        assert new_range > 0, "Range needs to be positive"

        self._im_arr = ( (self._im_arr - np.min(self._im_arr)) * new_range )/current_range + lower_limit

        return self._im_arr
    
    def set_windowing_parameters(self,
                sub_psf_window_sz: int,
                sigma: float, 
                neighborhood_size: int, 
                thresh_diff: float,
                prox_thresh: int, 
                 ):
        """
            Parameters:
            -----------
            sub_psf_window_sz: int
                Size of the square sub-PSF window in pixels
            sigma: float
                Standard deviation of the Gaussian filter
            neighborhood_size: int
                Pixel size of the region maximum/minimum filter is calculated on
            thresh_diff: float
                Threshold difference in intensity between to be considered a local maxima
            prox_thresh: int
                Maximum distance between two local maxima to be considered unique

            Returns:
            --------
            Input parameters in given order
        """
                
        self._sub_psf_window_sz = sub_psf_window_sz
        self._sigma = sigma
        self._neighborhood_size = neighborhood_size
        self._thresh_diff = thresh_diff
        self._prox_thresh = prox_thresh
        self._parameters_set = True

        return sub_psf_window_sz, sigma, neighborhood_size, thresh_diff, prox_thresh

    
    @staticmethod
    def filter_unique_coords(coords: np.array, prox_thresh: int):
        """
            Parameters:
            -----------
            coords: np.array
                2D array of pixel coordinates (unitless)
            prox_thresh: int
                Maximum distance between two coordinates to be considered unique.
                Any distance between two points that is less than this will be 
                considered as the same coordinate.

            Returns:
            --------
            np.array
                2D array of unique coordinates
        """
        local_maxima_idxs_filt = []
        for i, max_idx in enumerate(coords):

            # Check current coord isn't similar to any of the proceeding ones 
            if i < len(coords) - 1: 
                dist = np.linalg.norm(max_idx - coords[i+1:,None], axis = -1) # Distance between rest of locs
                repeat_bool = dist < prox_thresh
                if repeat_bool.sum() > 0:
                    # location is listed more than once - ignore current one
                    pass
                else:
                    local_maxima_idxs_filt.append(max_idx)

            else:
                # last point cannot be identical after filtering out the previous 
                # repeated 
                local_maxima_idxs_filt.append(max_idx)
        
        return np.array(local_maxima_idxs_filt)
    
    def get_local_maxima(self, sigma: float, neighborhood_size: int, thresh_diff: float,
                         prox_thresh: int, plot_blur = False):
        """
            Parameters:
            -----------
            sigma: float
                Standard deviation of the Gaussian filter
            neighborhood_size: int
                Pixel size of the region maximum filter is calculated on
            thresh_diff: float
                Threshold difference in intensity between local maxima and 
                minima
            prox_thresh: int
                Maximum distance between two local maxima to be considered unique

            Returns:
            --------
            np.array
                2D array of pixel coordinates of local maxima [row pixel idx, col pixel idx].
        """
        assert self._parameters_set == True, "Windowing parameters not set. Use set_windowing_parameters method"
        
        im_blur = filters.gaussian_filter(self._im_arr, sigma)

        if plot_blur:
            plt.imshow(im_blur, cmap='magma')
            plt.show()

        im_max = filters.maximum_filter(im_blur, neighborhood_size)
        local_maxima = (im_blur == im_max)
        im_min = filters.minimum_filter(im_blur, neighborhood_size)
        diff = ((im_max - im_min) > thresh_diff)
        local_maxima[diff == 0] = 0
        local_maxima_idxs = np.argwhere(local_maxima == 1)

        # Ensure there is no reptition in detection of same maxima
        local_maxima_idxs = self.filter_unique_coords(local_maxima_idxs, prox_thresh) 

        assert local_maxima_idxs.size != 0, """No sub-PSFs detected. Check windowing parameters.\n
        Max diff = {}""".format(np.max(im_max - im_min))

        return local_maxima_idxs

    def sub_psfs(self, plot_psf = False, plot_FT_psf = False, plot_blur = False):
        """
            Parameters:
            -----------
            plot_psf: bool
                If True, plots the sub-PSFs 
            plot_FT_psf: bool
                If True, plots the Fourier Transform of the sub-PSFs
            Returns:
            --------
            list, array, array
                - List of sub-PSF arrays (assumes square sub-PSF window size)
                - (n,4) array representing the pixel bounds to the sub-PSF arrays with rows of the 
                form: 
                [start_row, end_row, start_col, end_col], for n sub-psfs,
                - (n,2) representing the shift (x,y) = (col,row) (in pixels) to the center coord of the psf (0,0)
                NOTE: compatible with Wavefront propagate method definition of shift:
                     +ve pixel shift in x direction will shift center to left
                     +ve pixel shift in y direction will shift center down
        """
        assert self._parameters_set == True, "Windowing parameters not set. Use set_windowing_parameters method"

        local_maxima_idxs = self.get_local_maxima(self._sigma, self._neighborhood_size, self._thresh_diff, self._prox_thresh, plot_blur)
        half_window_size = int(self._sub_psf_window_sz/2)
        
        sub_psf_list = []  # store subpsf arrays
        sub_psf_bounds_list = [] # store subpsf bounds
        psf_center_shift_px = [] # store subpsf center shift in pixels
        for win_center in local_maxima_idxs:
            bounds = np.array([-half_window_size + win_center[0], half_window_size + win_center[0], -half_window_size + win_center[1], half_window_size + win_center[1]])
            sub_psf  = self._im_arr[bounds[0]:bounds[1], bounds[2]:bounds[3]]
            shift = np.array([-win_center[1] + self._psf_center_coord[1], -win_center[0] + self._psf_center_coord[0]]) #in [x,y] = [col,row]

            assert sub_psf.size != 0, "Cropping out of bounds. Reduce sub_psf_window_sz or increase image size."

            sub_psf_list.append(sub_psf)
            sub_psf_bounds_list.append(bounds)
            psf_center_shift_px.append(shift)

        if plot_psf:
            self.plot_sub_psfs(sub_psf_list, local_maxima_idxs)

        if plot_FT_psf: 
            self.plot_sub_fts(sub_psf_list, local_maxima_idxs)

        return sub_psf_list, np.asarray(sub_psf_bounds_list), np.asarray(psf_center_shift_px)

    def plot_sub_psfs(self, sub_psf_list, local_maxima_idxs):
        """
            Preface: i don't like functions that plot :( 
                     for debuggng/thresholding purposes ONLY
            Parameters:
            -----------
            sub_psf_list: list
                List of sub-PSF arrays
            local_maxima_idxs: np.array
                2D array of pixel coordinates of local maxima corresponding
                to the centers of the sub-PSF arrays. Ordered according to 
                sub_psf_list order.

            Returns:
            --------
            None
                Just plots
        """
        plt.figure()
        subplt_idx = 1
        for i in range(len(sub_psf_list)):
            plt.subplot(1, len(sub_psf_list), subplt_idx)
            plt.imshow(sub_psf_list[i], cmap='magma')
            ax = plt.gca()
            ax.set_title(local_maxima_idxs[i])
            subplt_idx += 1
        
        plt.show()

        return None
    
    def plot_ft(self):
        """
            TODO for single PSF with other input parameters in future
        """

        pass

    def plot_sub_fts(self, sub_psf_list, local_maxima_idxs):
        """
            Preface: i don't like functions that plot :( 
                     for debuggng/thresholding purposes ONLY
            Parameters:
            -----------
            sub_psf_list: list
                List of sub-PSF arrays
            local_maxima_idxs: np.array
                2D array of pixel coordinates of local maxima corresponding
                to the centers of the sub-PSF arrays. Ordered according to 
                sub_psf_list order.

            Returns:
            --------
            None
                Just plots
        """
        subplt_idx = 1
        plt.figure()
        for i in range(len(sub_psf_list)):
            plt.subplot(1, len(sub_psf_list), subplt_idx)

            im_fft = np.abs(np.fft.fftshift(np.fft.fft2(sub_psf_list[i])))**2 # fftshift to shift 0-freq component to center, fft2 to compute 2D FT
            plt.imshow(np.log10(im_fft), cmap='magma')
            ax = plt.gca()
            ax.set_title(local_maxima_idxs[i])

            subplt_idx += 1

        plt.show()

        return None
