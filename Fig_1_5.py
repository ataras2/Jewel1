"""
    Plot for JATIS Paper relating to simulated (ideal) jewel optical system with real jewel lab data
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import PowerNorm
import ehtplot.color

from skimage.io import imread
from skimage.filters import window
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
from skimage import draw

import dLux as dl
import dLux.utils as dlu

from Jewels import MaskPattern, JewelMask, JewelAngularOpticalSystem
from Materials import Material
from Telescopes import telescope_factory
from Psf import PSF

plt.rcParams["image.origin"] = 'lower'


# ------- Physical Parameters ---------------------------------------------------------------------#
# Telescope (VAMPIRES instrument)
tscope = telescope_factory("subaru_vis")

# Jewel Mask (in pupil wheel)
prim_diam = tscope.pupil         # Primary diameter (m)
grid_size = 7e-03                # Dimension of underlying rectangular grid within primary diam (m)
shim_width = 140e-6               # Width of the shim (m)

# Jewel Mask simulated
n_pix = 512                                        # Number of pixels defining primary diameter
pixel_coords = dlu.pixel_coords(n_pix, prim_diam)  # Pixel coordinates for pixel centers defining entire mask

# Wavefront simulated (point source)
wf_diam = prim_diam             # Diameter of initial wavefront to propagate wavefront (m)
wf_npix = n_pix                 # Number of pixels defining wavefront
wavelength_center = 635e-09     # Central wavelength in band (m)

# Detector
psf_npix = 100                  # Number of pixels along one dim of the PSF/oversample 
oversample = 40                 # Oversampling factor for the PSF
psf_pixel_scale = 1.238 * oversample # arcsec/px

# Wedge deviation
mat = Material("mgf2")
wedge_angle = 80 * 1/60 #30*1/60  #deg
expected_dev = (wedge_angle*60) * (mat.refractive_index(wavelength_center)-1.0)

# -------------------------------------------------------------------------------------------------#
# Generate MaskPattern from IDL .dat file
fname = "data/find18_4x4h_mod.dat" #"data/find20_4x5h.dat"#
sol_idx = 10#18 
pattern_tf = np.array([0,0, np.pi/180 *0])                   # [x_translation (m), y_translation (m), rotation (rad)]
manual_machining_sequence = [np.array([0, 1]), np.array([0, 2])] # pattern idxs for wedge i, j etc
# manual_machining_sequence = [np.array([0, 2]), np.array([0, 1])] # pattern idxs for wedge i, j etc
mask_pattern = MaskPattern.from_file(fname=fname, 
                                    solution_idx=sol_idx, 
                                    primary_diam=prim_diam, 
                                    grid_size=grid_size, 
                                    shim_width=shim_width, 
                                    tf=pattern_tf,
                                    manual_machining_seq=manual_machining_sequence,
                                    )

# Data is with masked flip so flip position of centers.
cart_centers = mask_pattern.cart_hex_centers
flipped_centers = []
for pattern_i_hex_centers in cart_centers:
    # flip along y-axis
    new_pattern = []
    for cen in pattern_i_hex_centers:
        new_pattern.append([cen[0], -cen[1]])

    flipped_centers.append(np.asarray(new_pattern))

flipped_centers = np.asarray(flipped_centers)
mask_pattern.set_cart_hex_centers(flipped_centers)

# -------------------------------------------------------------------------------------------------#
# Generate JewelMask and JewelMask optical system
jewel_mask = JewelMask(mask_pattern=mask_pattern, 
                       n_pix=n_pix, 
                       pixel_coords=pixel_coords, 
                       wedge_angles=np.array([wedge_angle, wedge_angle]),
                       slope_orientations= np.array([np.pi/180 *0 , np.pi/180 *90]), 
                       materials=[mat, mat],
                       max_machining_err=None,
                    ) 
# Create dLux-based optical system 
optics = JewelAngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers = jewel_mask.jewel_layers,
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
    psf_shifts=None, # To set later
    sub_psf_npixels = None # ""
)
# -------------------------------------------------------------------------------------------------#
# Lab data
fname = "data/4_4_glued_002_mean_image.png"

data_psf = imread(fname, as_gray=True) 

data_PSF = PSF(data_psf)
sub_psf_window_sz = 500
data_PSF.set_windowing_parameters(sub_psf_window_sz= sub_psf_window_sz,  
                                  sigma= 20,           
                                  neighborhood_size= 10,   
                                  thresh_diff= 0.003,         
                                  prox_thresh= 5,         
                                  )         
data_psfs_list, data_psf_bounds, data_shifts = data_PSF.sub_psfs(plot_psf = False, plot_FT_psf = False, plot_blur=False)
# -------------------------------------------------------------------------------------------------#
# Simulated  data
simu_psf = optics.propagate_mono(wavelength_center)
simu_PSF = PSF(simu_psf)
simu_PSF.scale_intensity(upper_limit=data_psf.max(), lower_limit=data_psf.min())  # Scale intensity with data
simu_psf = simu_PSF._im_arr
sub_psf_window_sz = 500  #(px, assuming square window)
simu_PSF.set_windowing_parameters(sub_psf_window_sz= sub_psf_window_sz,  
                                  sigma= 15,           
                                  neighborhood_size= 10,   
                                  thresh_diff= 0.003,         
                                  prox_thresh= 10,         
                                  )         
simu_sub_psfs_list, simu_sub_psf_bounds, simu_shifts = simu_PSF.sub_psfs(plot_psf = False, plot_FT_psf = False, plot_blur=False)

# -------------------------------------------------------------------------------------------------#
# Function for compuing power spectra and splodges mask
def calc_bispectrum(im_array, im_gauss_std):
    """
        Parameters:
        ------------
        im_array: np.array
            2D square image array
        im_gauss_std: int
            Standard deviation of Gaussian window to apply to image (prior to taking fft)
        Returns:
        _________
        bispectrum: np.array
            Returns complex visibility function (/spatial spectrum)
        power_spectrum_processed: np.array
            Power spectra (or square mod of bispectrum), post gaussian windowing
            and power scaling for visualsiation
        phase: np.array
            Complex visbility component (i.e. phase of bispectrum)
        
    """
    sz = im_array.shape[0] 
    gauss_window_im = window(window_type=("gaussian", im_gauss_std), shape = (sz,sz))

    im_array = im_array * gauss_window_im
    bispectrum = np.fft.fftshift(np.fft.fft2(im_array))

    power_spectrum = np.abs(bispectrum)**2
    phase = np.angle(bispectrum)

    power_spectrum_processed = power_spectrum

    return bispectrum, power_spectrum_processed, phase

def get_splodges_mask(im_array, min_peak, radius, prox_thresh = 1):
    """
        Parameters:
        ------------
        im_array: np.array
            2D square image array
        min_peak: float
            Minimum intensity value for a peak (/local maxima) to be considered
        radius: int
            Radius of splodges in pixels
        prox_thresh: int
            Minimal distance allowed separating peaks
            
        Returns:
        _________
        splodge_mask: np.array
            Binary mask of splodges in image (1 = splodge, 0 = no splodge)
        
    """
    splodge_coords = feature.peak_local_max(np.asarray(im_array), threshold_abs=min_peak, min_distance=prox_thresh) 
    assert len(splodge_coords) > 0 and len(splodge_coords) < 50, "None or too many splodges"

    # # Draw mask 
    splodge_mask = np.zeros(im_array.shape)
    for cen in splodge_coords:
        rr, cc = draw.disk((cen[0], cen[1]), radius)
        splodge_mask[rr, cc] = 1.0
    
    return splodge_mask
# -------------------------------------------------------------------------------------------------#
# Plotting (+ OTF calcs)
mosaic = """
    AA.JJ
    AA.JJ
    BC.KL
    DE.MN
    FG.OP
    HI.QR
    """
fig = plt.figure(constrained_layout=True, figsize=(6.3, 6.67))
axes = fig.subplot_mosaic(mosaic,gridspec_kw={"wspace": 0.0,"hspace": 0.0})  

intensity_max, intensity_min = data_psf.max(), data_psf.min()

sorted_simu_idx = [3, 1, 2, 0]
simu_sub_psfs_list = [simu_sub_psfs_list[i] for i in sorted_simu_idx]
simu_sub_psfs_list = [ndi.rotate(psf_im, -90, reshape=False) for psf_im in simu_sub_psfs_list]
sub_psfs = [simu_sub_psfs_list, data_psfs_list] 

axes_alph = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R"]
psfs = [simu_psf, data_psf]
psf_centers = [[1258,1258],[1250,1500]] # px
cropped_psf_sz = [1258*2,2500] 
fft_sz = 40 # px
sub_psf_plt_size = [170,170]
fov = cropped_psf_sz[0]*psf_pixel_scale/oversample * 1/tscope.magnification # detector fov (arsec, on-sky) gives same result as calling from wavefront.diameter

BFS_px_sep = 2.4e-6 *1e3 # (mm)
f_det = 400 # Focal length from lens to detector (mm)
px_ang_sep = 2*np.arctan( (BFS_px_sep/2)/f_det ) # (rad)
det_FOV = dlu.rad2arcsec(px_ang_sep * cropped_psf_sz[1] * 1/tscope.magnification) # (arsec, on-sky)

extents = [[-fov/2, fov/2, -fov/2, fov/2], [-det_FOV/2, det_FOV/2, -det_FOV/2, det_FOV/2]]

subplt_it = 0
max_power, min_power = [],[] # for colorbar scaling
vmax_power, vmin_power = 2e6, 0 #3.37, 0.059#to set after first run
norm_ps = PowerNorm(0.08, vmax=vmax_power, vmin=vmin_power)
# norm_psf =  PowerNorm(0.3, vmax=intensity_max, vmin=intensity_min)
norm_psf =  PowerNorm(0.5, vmax=intensity_max, vmin=intensity_min)
for i, psf in enumerate(psfs):
    # Cropped whole psf
    hlf_sz = int(cropped_psf_sz[i]/2)
    psf_im = psf[psf_centers[i][0] - hlf_sz:psf_centers[i][0] + hlf_sz,
                    psf_centers[i][1] - hlf_sz:psf_centers[i][1] + hlf_sz]
    
    if i == 0:
        psf_im  = ndi.rotate(psf_im, -90, reshape=False)
        axes[axes_alph[subplt_it]].set_title("Simulated")
        axes[axes_alph[subplt_it]].set_ylabel("Y (arcsec)")
    else:
        axes[axes_alph[subplt_it]].set_title("Recorded")

    axes[axes_alph[subplt_it]].set_xlabel("X (arcsec)")
    axes[axes_alph[subplt_it]].set_xlim([-1.6,1.6])
    axes[axes_alph[subplt_it]].set_ylim([-1.6,1.6])
    im1 = axes[axes_alph[subplt_it]].imshow(psf_im, cmap='bone_ur', extent = extents[i], norm = norm_psf) 
    subplt_it += 1

    # Sub-psfs
    for j in range(4):
        sub_psf = sub_psfs[i][j]
        middle = int(sub_psf.shape[0]/2)
        axes[axes_alph[subplt_it]].imshow(sub_psf, cmap='bone_ur', norm = norm_psf) 
        axes[axes_alph[subplt_it]].set_xticks([])
        axes[axes_alph[subplt_it]].set_yticks([])

        axes[axes_alph[subplt_it]].set_xlim([middle - sub_psf_plt_size[i], middle + sub_psf_plt_size[i]])
        axes[axes_alph[subplt_it]].set_ylim([middle - sub_psf_plt_size[i], middle + sub_psf_plt_size[i]])

        subplt_it += 1

    # Power spectra
    for j in range(4):
        sub_psf = sub_psfs[i][j]
        if i == 1:
            sub_psf = sub_psf**1.2 # non-linearity of detector approximation

        bispectrum, power_spectrum, phase = calc_bispectrum(im_array=sub_psf, im_gauss_std=150)
        middle = int(power_spectrum.shape[0]/2)
        power_spectrum = power_spectrum[middle-fft_sz:middle+fft_sz , middle-fft_sz:middle+fft_sz]
        sploge_mask = get_splodges_mask(power_spectrum, min_peak = 2e4, radius=5)
        im = power_spectrum*sploge_mask
        max_power.append(im.max())
        min_power.append(im.min())

        im2 = axes[axes_alph[subplt_it]].imshow(im, cmap='pink_ur', norm=norm_ps) 
        axes[axes_alph[subplt_it]].set_xticks([])
        axes[axes_alph[subplt_it]].set_yticks([])
        subplt_it += 1

    
vmax_power, vmin_power = np.array(max_power).max(), np.array(min_power).min()

scalarmappable = cm.ScalarMappable(norm=norm_psf, cmap="magma")
scalarmappable.set_clim(vmax=intensity_max, vmin=intensity_min)
fig.colorbar(im1, ax =[axes["J"], axes["L"], axes["N"]] , label = "Intensity", aspect = 30)

scalarmappable = cm.ScalarMappable(norm=norm_ps, cmap="pink_ur")
scalarmappable.set_clim(vmin = vmin_power, vmax = vmax_power)
fig.colorbar(scalarmappable, ax =[ axes["P"],axes["R"] ], label = "Power", pad = 0.15, aspect = 10)
plt.show()

# -------------------------------------------------------------------------------------------------#
# E.g of 9 inteferogram pattern
fname = "data/VC7x6s.txt"
sol_idx = 0
pattern_tf = np.array([0.23e-3,-0.1e-3, np.pi]) # [x_translation (m), y_translation (m), rotation (rad)]
mask_pattern = MaskPattern.from_file(fname=fname, 
                                    solution_idx=sol_idx, 
                                    primary_diam=prim_diam, 
                                    secondary_diam=tscope.secondary / (tscope.primary/tscope.pupil),
                                    grid_size=grid_size, 
                                    shim_width=shim_width, 
                                    tf=pattern_tf,
                                    # manual_machining_seq=manual_machining_sequence,
                                    )

wedge_angle = wedge_angle/2
jewel_mask = JewelMask(mask_pattern=mask_pattern, 
                       n_pix=n_pix, 
                       pixel_coords=pixel_coords, 
                       wedge_angles=np.array([wedge_angle, wedge_angle, wedge_angle]),
                       slope_orientations= np.array([np.pi/180 *0 , np.pi/180 *120, np.pi/180 *240]), 
                       materials=[mat, mat, mat],
                       max_machining_err=None,
                    ) 
# Create dLux-based optical system 
psf_pixel_scale  = psf_pixel_scale*0.6
optics = JewelAngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers = jewel_mask.jewel_layers,
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
    psf_shifts=None, # To set later
    sub_psf_npixels = None # ""
)

psf = optics.propagate_mono(wavelength_center)
norm_psf =  PowerNorm(0.5, vmax=psf.max(), vmin=psf.min())
plt.imshow(psf, norm = norm_psf, cmap = 'bone_ur')
plt.colorbar(label="Power")

simu_PSF = PSF(psf)
sub_psf_window_sz = 1000  #(px, assuming square window)
simu_PSF.set_windowing_parameters(sub_psf_window_sz= sub_psf_window_sz,  
                                  sigma= 15,           
                                  neighborhood_size= 10,   
                                  thresh_diff= 2.5e-13,         
                                  prox_thresh= 10,         
                                  )         
simu_sub_psfs_list, simu_sub_psf_bounds, simu_shifts = simu_PSF.sub_psfs(plot_psf = False, plot_FT_psf = False, plot_blur=False)

mosaic = """
        .AB.
        .CDE
        .FG.
        """
alph = ["F", "G", "C", "E", "D", "B", "A"]
fig = plt.figure(constrained_layout=True, figsize=(6,6))
axes = fig.subplot_mosaic(mosaic,gridspec_kw={"wspace": 0.0,"hspace": 0.0})  

fft_sz = 45 # px
max_power, min_power = [],[] # for colorbar scaling
vmin_power,vmax_power = 4.5e-31, 2.6e-14 
norm_ps = PowerNorm(0.08, vmax=vmax_power, vmin=vmin_power)
for i, ax in enumerate(alph):
    # Power spectra
    sub_psf = simu_sub_psfs_list[i]
    sub_psf = np.pad(sub_psf, 100)
    bispectrum, power_spectrum, phase = calc_bispectrum(im_array=sub_psf, im_gauss_std=200)
    middle = int(power_spectrum.shape[0]/2)
    power_spectrum = power_spectrum[middle-fft_sz:middle+fft_sz , middle-fft_sz:middle+fft_sz]
    sploge_mask = get_splodges_mask(power_spectrum, min_peak = 2e-16, radius=5)
    im = power_spectrum*sploge_mask
    max_power.append(im.max())
    min_power.append(im.min())

    im2 = axes[ax].imshow(im, cmap='pink_ur', norm=norm_ps) 
    axes[ax].set_xticks([])
    axes[ax].set_yticks([])

vmax_power, vmin_power = np.array(max_power).max(), np.array(min_power).min() # to determine limits with

scalarmappable = cm.ScalarMappable(norm=norm_ps, cmap="pink_ur")
scalarmappable.set_clim(vmin = vmin_power, vmax = vmax_power)
fig.colorbar(scalarmappable, ax =[ axes["B"],axes["E"],axes["G"] ], label = "Power", pad = 0.15, aspect = 15)

pat_0_cent = mask_pattern.cart_hex_centers[2,:,:]
sub_ap  = []
for cen in pat_0_cent:
    tf = dl.CoordTransform(translation=cen, rotation=np.pi/6)
    sub_ap.append(dl.RegPolyAperture(nsides=6, rmax=mask_pattern._rmax, transformation=tf))

aperture = dl.MultiAperture(sub_ap)
trans = aperture.transmission(pixel_coords, prim_diam/n_pix)
optics = JewelAngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers = [("aperture", dl.TransmissiveLayer(transmission=trans))],
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
    psf_shifts=None, # To set later
    sub_psf_npixels = None # ""
)
psf = optics.propagate_mono(wavelength_center)
norm_psf =  PowerNorm(0.5, vmax=psf.max(), vmin=psf.min())
plt.figure()
plt.imshow(psf, norm = norm_psf, cmap = 'bone_ur')
plt.colorbar(label="Power")

plt.figure()
fft_sz = 200
middle = int(psf.shape[0]/2)
psf = np.pad(psf, 100)
psf_sz = 500
sub_psf = psf[middle-psf_sz:middle+psf_sz , middle-psf_sz:middle+psf_sz]
bispectrum, power_spectrum, phase = calc_bispectrum(im_array=psf, im_gauss_std=200)
middle = int(power_spectrum.shape[0]/2)
power_spectrum = power_spectrum[middle-fft_sz:middle+fft_sz , middle-fft_sz:middle+fft_sz]
sploge_mask = get_splodges_mask(power_spectrum, min_peak = 2e-16, radius=16)
im = power_spectrum*sploge_mask
plt.imshow(im, cmap='pink_ur', norm=norm_ps) 
plt.show()
