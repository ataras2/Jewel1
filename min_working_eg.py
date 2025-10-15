"""
    7x6 min working eg
"""

import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import PowerNorm

from Jewels import MaskPattern, JewelMask
from Materials import Material
from Telescopes import telescope_factory
from Psf import PSF

from scipy import ndimage as ndi
from skimage import feature
from skimage import draw

# ------- Physical Parameters ---------------------------------------------------------------------#
# Telescope (VAMPIRES instrument on Subaru Telescope)
tscope = telescope_factory("subaru_vis")

# Jewel Mask (in pupil wheel)
prim_diam = tscope.pupil         # Primary diameter of pupil (m)
grid_size = 7e-03                 # Dimension of underlying rectangular grid within primary diam (m)
shim_width = 90e-6               # Width of the shim (m) 

# Jewel Mask simulated
n_pix = 512                                        # Number of pixels defining primary diameter
pixel_coords = dlu.pixel_coords(n_pix, prim_diam)  # Pixel coordinates for pixel centers defining primary mask

# Wavefront simulated (point source)
wf_diam = prim_diam         # Diameter of initial wavefront to propagate wavefront (m)
wf_npix = n_pix                 # Number of pixels defining wavefront
wavelength_center = 750e-09     # Central wavelength in band (m)

# Detector
psf_npix = 512               # Number of pixels along one dim of the PSF/oversample 
psf_pixel_scale = 6e-3*tscope.magnification     # arsec on det 
oversample = 1             # Oversampling factor for the PSF

# Wedge deviation
mat = Material("sio2")
desired_dev_sky = 1 # (arcsec, on sky)
desired_dev_det = desired_dev_sky * tscope.primary/tscope.pupil  * 1/60 # arcmin, on detector
wedge_angle = desired_dev_det/(mat.refractive_index(wavelength_center)-1) * 1/60 # (deg)
print("Wedge angle: {} deg, {} arcmin".format(wedge_angle, wedge_angle*60))

# -------------------------------------------------------------------------------------------------#
# Generate MaskPattern from IDL .dat file
fname = "data/VC7x6s.txt"
sol_idx = 0 
pattern_tf = np.array([0.23e-3, -0.17e-3, np.pi/180*0])  # [x_translation (m), y_translation (m), rotation (rad)]
mask_pattern = MaskPattern.from_file(fname=fname, 
                                    solution_idx=sol_idx, 
                                    primary_diam=prim_diam, 
                                    grid_size=grid_size, 
                                    # manual_machining_seq=manual_machining_sequence,
                                    shim_width=shim_width, 
                                    tf=pattern_tf,
                                    )
mask_pattern.display_pattern()
print("Machining seq: {}".format(mask_pattern.machining_sequence))
print("Rmax:", mask_pattern._rmax)
plt.show()

# -------------------------------------------------------------------------------------------------#
# Generate JewelMask and JewelMask optical system
jewel_mask = JewelMask(mask_pattern=mask_pattern, 
                       n_pix=n_pix, 
                       pixel_coords=pixel_coords, 
                       wedge_angles=np.array([wedge_angle, wedge_angle, wedge_angle]),
                       slope_orientations= np.array([np.pi/180 *0 , np.pi/180 *120, np.pi/180 *240]), 
                       materials=[mat, mat, mat],
                    #    glass_trans = np.array([transmittance]),
                       max_machining_err=None,
                    ) 
# -------------------------------------------------------------------------------------------------#
# Create dLux-based optical system 
optics = dl.AngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers=jewel_mask.jewel_layers,
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
)
# -------------------------------------------------------------------------------------------------#
mono_psf = optics.propagate_mono(wavelength=wavelength_center)

chrom_psf = optics.propagate(wavelengths=np.array([650e-9, 750e-9, 850e-9]), weights=np.array([1,1,1]))
plt.imshow(chrom_psf)
plt.colorbar()
plt.plot()
plt.show()