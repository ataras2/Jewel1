"""
    Script to compare the throughput of a Jewel mask with a regular opaque aperture mask as
    well as demonstrate binary search for Jewel mask tesselations.
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
from skimage.filters import window
import ehtplot.color
from skimage import feature
from skimage import draw


plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 72
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.titlesize"] = 20

# ------- Physical Parameters ---------------------------------------------------------------------#
# Telescope (VAMPIRES instrument)
tscope = telescope_factory("subaru_vis")

# Jewel Mask (in pupil wheel)
prim_diam = tscope.pupil  # Primary diameter (m)
secondary_diam = (
    tscope.secondary / tscope.primary * tscope.pupil
)  # Secondary diameter (m)
grid_size = 7e-03  # Dimension of underlying rectangular grid within primary diam (m)
shim_width = 140e-6  # Width of the shim (m)

# Jewel Mask simulated
n_pix = 512  # Number of pixels defining primary diameter
pixel_coords = dlu.pixel_coords(
    n_pix, prim_diam
)  # Pixel coordinates for pixel centers defining primary mask

# Wavefront simulated (point source)
wf_diam = prim_diam * 1.2  # Diameter of initial wavefront to propagate wavefront (m)
wf_npix = n_pix  # Number of pixels defining wavefront
wavelength_center = 650e-09  # Central wavelength in band (m)

# Detector
psf_npix = 64  # Number of pixels along one dim of the PSF/oversample
psf_pixel_scale = 300e-1  # arsec (decreasing provides greater mag)
oversample = 32  # Oversampling factor for the PSF

# Wedge deviation
mat = Material("mgf2")
dev_scale = 0.7  # to fit on detector for psf visibility without having oversample too much for resolution
wedge_angle = 30 * 1 / 60 * dev_scale  # deg

reflectance = 0.024548  # refractiveindex.info
transmittance = 1 - reflectance

# -------------------------------------------------------------------------------------------------#
# Generate tscope aperture layer (const between two optical systems) and get initial intensity

primary = dlu.circle(coords=pixel_coords, radius=0.5 * prim_diam)
secondary = dlu.circle(coords=pixel_coords, radius=0.5 * secondary_diam)
tscope_ap = primary - secondary
tscope_layer_tup = ("tscope", dl.layers.TransmissiveLayer(transmission=tscope_ap))
tscope_optic = dl.AngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers=[tscope_layer_tup],
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
)
tscope_psf = tscope_optic.propagate_mono(wavelength_center)
init_intensity = tscope_psf.sum()
# -------------------------------------------------------------------------------------------------#
# Generate MaskPattern from IDL .dat file
fname = "data/find18_4x4h_mod.dat"  # "data/find20_4x5h.dat"#
sol_idx = 10  # 18
pattern_tf = np.array(
    [0, 0, np.pi / 180 * 60]
)  # [x_translation (m), y_translation (m), rotation (rad)]
manual_machining_sequence = [
    np.array([0, 1]),
    np.array([0, 2]),
]  # pattern idxs for wedge i, j etc
mask_pattern = MaskPattern.from_file(
    fname=fname,
    solution_idx=sol_idx,
    primary_diam=prim_diam,
    grid_size=grid_size,
    shim_width=shim_width,
    tf=pattern_tf,
    manual_machining_seq=manual_machining_sequence,
)

# For later plotting to illustrate binary search for Jewel tesselations
hex_idxs = np.array([[23, 2, 19, 16], [11, 18, 0, 14], [1, 8, 24, 5], [13, 6, 10, 22]])
dummy_mask_pattern = MaskPattern(
    n_pat=4,
    n_seg=4,
    hex_idxs=hex_idxs,
    idim=5,
    primary_diam=prim_diam,
    grid_size=grid_size,
    shim_width=shim_width,
    tf=pattern_tf,
    manual_machining_seq=manual_machining_sequence,
)

# -------------------------------------------------------------------------------------------------#
# Generate JewelMask and JewelMask optical system
jewel_mask = JewelMask(
    mask_pattern=mask_pattern,
    n_pix=n_pix,
    pixel_coords=pixel_coords,
    wedge_angles=np.array([wedge_angle, wedge_angle]),
    slope_orientations=np.array([np.pi / 180 * 45, np.pi / 180 * 135]),
    materials=[mat, mat],
    glass_trans=1 - 2 * reflectance,
)
optics = dl.AngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers=jewel_mask.jewel_layers,
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
)

optics = optics.insert_layer(layer=tscope_layer_tup, index=0)

# PSF
wf = optics.propagate_mono(wavelength_center, return_wf=True)
jwl_psf = wf.psf

# --- affine tf to look similar to data (purely illustrative)
psf = jwl_psf[280:1150, 600:1450]
# plt.imshow(psf)
# plt.show()
psf = np.pad(psf, pad_width=5, constant_values=0)
theta = np.pi / 180 * 45
w, h = psf.shape
mat_reflect = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ np.array(
    [[1, 0, 0], [0, 1, -h], [0, 0, 1]]
)
psf = ndi.affine_transform(psf, mat_reflect)
mat_rotate = (
    np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]])
    @ np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [np.sin(theta), -np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    @ np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
)
psf = ndi.affine_transform(psf, mat_rotate)


simu_PSF = PSF(jwl_psf)
sub_psf_window_sz = 500  # (px, assuming square window)
simu_PSF.set_windowing_parameters(
    sub_psf_window_sz=sub_psf_window_sz,
    sigma=15,
    neighborhood_size=10,
    thresh_diff=0.1e-11,
    prox_thresh=10,
)
simu_sub_psfs_list, simu_sub_psf_bounds, simu_shifts = simu_PSF.sub_psfs(
    plot_psf=False, plot_FT_psf=False, plot_blur=False
)

# -------------------------------------------------------------------------------------------------#
# Jewel Pattern optimisation illustration
colour_vect = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
centers = [dummy_mask_pattern._cart_hex_centers, mask_pattern._cart_hex_centers]
titles = ["Initial Tiling", "Tile Swap"]
xlabels = ["(a)", "(b)"]
plt.figure(figsize=(12, 5))
subplt_it = 1
for j, cart_hex_centers in enumerate(centers):
    plt.subplot(1, 2, subplt_it)
    ax = plt.gca()
    lgnd_patches = []
    for i, pattern in enumerate(cart_hex_centers):
        patch = 0
        for k, seg_coord in enumerate(pattern):
            if i == 3 and k == 3:
                line_width = 2
                edge_color = "black"
            elif i == 2 and k == 2:
                line_width = 2
                edge_color = "black"
            else:
                line_width = 1
                edge_color = colour_vect[i]

            patch = patches.RegularPolygon(
                (seg_coord[0], seg_coord[1]),
                6,
                radius=mask_pattern._rmax,  #  color=colour_vect[i]
                label=str(i),
                orientation=-mask_pattern.seg_rotation + np.pi / 6,
                edgecolor=edge_color,
                facecolor=colour_vect[i],
                linewidth=line_width,
            )
            ax.add_artist(patch)
        lgnd_patches.append(patch)

        primary_m = plt.Circle((0.0, 0.0), prim_diam / 2.0, color="black", fill=False)
        ax.add_artist(primary_m)
        cen = plt.scatter(0, 0, s=1)
        ax.add_artist(cen)
        ax.set(
            xlim=(-prim_diam * 1.1 / 2, prim_diam * 1.1 / 2),
            ylim=(-prim_diam * 1.1 / 2, prim_diam * 1.1 / 2),
        )

        plt.title(titles[j])
        plt.xlabel(xlabels[j])
        plt.xticks([])
        plt.yticks([])
        plt.legend(
            handles=lgnd_patches,
            labels=["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4"],
            loc="upper right",
        )

    subplt_it += 1


# Plot difference in Fourier space between patterns
def get_splodges_mask(im_array, min_peak, radius, prox_thresh=1):
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
    splodge_coords = feature.peak_local_max(
        onp.asarray(im_array), threshold_abs=min_peak, min_distance=prox_thresh
    )
    assert (
        len(splodge_coords) > 0 and len(splodge_coords) < 50
    ), "None or too many splodges"

    # # Draw mask
    splodge_mask = onp.zeros(im_array.shape) + np.nan
    for cen in splodge_coords:
        rr, cc = draw.disk((cen[0], cen[1]), radius)
        splodge_mask[rr, cc] = 1.0

    return splodge_mask


dummy_jewel_mask = JewelMask(
    mask_pattern=dummy_mask_pattern,
    n_pix=n_pix,
    pixel_coords=pixel_coords,
    wedge_angles=np.array([wedge_angle, wedge_angle]),
    slope_orientations=np.array([np.pi / 180 * 45, np.pi / 180 * 135]),
    materials=[mat, mat],
    #    glass_trans = transmittance,
)
optics = dl.AngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers=dummy_jewel_mask.jewel_layers,
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
)

optics = optics.insert_layer(layer=tscope_layer_tup, index=0)

dummy_psf = optics.propagate_mono(wavelength_center)
dummy_psf = PSF(dummy_psf)
dummy_psf.set_windowing_parameters(
    sub_psf_window_sz=sub_psf_window_sz,
    sigma=15,
    neighborhood_size=10,
    thresh_diff=0.1e-11,
    prox_thresh=10,
)

dummy_sub_psfs_list, dummy_sub_psf_bounds, dummy_shifts = dummy_psf.sub_psfs(
    plot_psf=False, plot_FT_psf=False
)


dummy_psf = np.pad(dummy_sub_psfs_list[0], 200)
opt_psf = np.pad(simu_sub_psfs_list[0], 200)

ax_lim = [opt_psf.shape[0] / 2 - 60, opt_psf.shape[0] / 2 + 60]

plt.figure(figsize=(12, 5))
crop_lim = [50, 100]
plt.subplot(1, 2, 1)
bkgnd_cutoff = 0.027
im_fft = np.abs(np.fft.fftshift(np.fft.fft2(dummy_psf))) ** 2
norm_ps = PowerNorm(0.25, vmax=im_fft.max(), vmin=im_fft.min())
plt.xlim(ax_lim)
plt.ylim(ax_lim)
plt.xticks([])
plt.yticks([])
plt.xlabel("(a)")
plt.suptitle("Pattern 4 Interferogram")
plt.title("Initial Tiling")
processed_fft = im_fft
splodges_mask = get_splodges_mask(processed_fft, min_peak=5e-15, radius=8)
plt.imshow(processed_fft * splodges_mask, cmap="pink_ur", norm=norm_ps)

plt.subplot(1, 2, 2)
im_fft = np.abs(np.fft.fftshift(np.fft.fft2(opt_psf))) ** 2
plt.xlim(ax_lim)
plt.ylim(ax_lim)
plt.xticks([])
plt.yticks([])
plt.title("Tile Swap")
plt.xlabel("(b)")
processed_fft = im_fft  # **0.1
splodges_mask = get_splodges_mask(processed_fft, min_peak=5e-15, radius=8)
plt.imshow(processed_fft * splodges_mask, cmap="pink_ur", norm=norm_ps)
plt.colorbar(label="Power", norm=norm_ps)

plt.show()
# -------------------------------------------------------------------------------------------------#
# Generate opaque mask with JewelMask pattern
# For one NR pattern
ideal_mask_pattern = MaskPattern.from_file(
    fname=fname,
    solution_idx=sol_idx,
    primary_diam=prim_diam,
    grid_size=grid_size,
    shim_width=0,  # an ideal conventional mask doesn't need shim
    tf=pattern_tf,
    manual_machining_seq=manual_machining_sequence,
)
single_pat_cens = ideal_mask_pattern.cart_hex_centers[
    1
]  # just grabbing first (no justification really)

sub_ap = []
for cen in single_pat_cens:
    tf = dl.CoordTransform(translation=cen, rotation=np.pi / 6)
    sub_ap.append(
        dl.RegPolyAperture(nsides=6, rmax=ideal_mask_pattern._rmax, transformation=tf)
    )

aperture = dl.MultiAperture(sub_ap)
trans = aperture.transmission(pixel_coords, prim_diam / n_pix)

layers = [tscope_layer_tup, ("aperture", dl.TransmissiveLayer(transmission=trans))]
opaque_optic = dl.AngularOpticalSystem(
    wf_npixels=wf_npix,
    diameter=wf_diam,
    layers=layers,
    psf_npixels=psf_npix,
    psf_pixel_scale=psf_pixel_scale,
    oversample=oversample,
)
trans = opaque_optic.aperture.transmission
op_psf = opaque_optic.propagate_mono(wavelength_center)


def calc_jewel_throughput(n_surfs, reflec):

    jewel_mask = JewelMask(
        mask_pattern=mask_pattern,
        n_pix=n_pix,
        pixel_coords=pixel_coords,
        wedge_angles=np.array([wedge_angle, wedge_angle]),
        slope_orientations=np.array([np.pi / 180 * 45, np.pi / 180 * 135]),
        materials=[mat, mat],
        # glass_trans=1 - n_surfs * reflec,
        glass_trans=(1 - reflec) ** n_surfs,
    )
    optics = dl.AngularOpticalSystem(
        wf_npixels=wf_npix,
        diameter=wf_diam,
        layers=jewel_mask.jewel_layers,
        psf_npixels=psf_npix,
        psf_pixel_scale=psf_pixel_scale,
        oversample=oversample,
    )

    optics = optics.insert_layer(layer=tscope_layer_tup, index=0)

    # PSF
    wf = optics.propagate_mono(wavelength_center, return_wf=True)
    jwl_psf = wf.psf
    return float(jwl_psf.sum() / init_intensity * 100)


# -------------------------------------------------------------------------------------------------#
# Compare total intensities
OM_dI = float(op_psf.sum() / init_intensity * 100)
JM_dI = calc_jewel_throughput(2, reflectance)

print("Percentage transmission of incoming light:")
print(f"Jewel mask: %{round(JM_dI, 2)} \nOpaque mask %{round(OM_dI, 2)}")
print(
    f"Jewel mask transmits {round(float(JM_dI / OM_dI), 2)} times greater total intensity than opaque mask"
)


# now do cases
n_g = 1.375
n_a = 1.0
reflect_no_ar = ((n_g - n_a) / (n_g + n_a)) ** 2
reflect_ar = 0.7 / 100
n_surfs = [2, 4]  # glued stack, dry stack
reflec = [reflect_no_ar, reflect_ar]

import itertools

for n_surf, ref in itertools.product(n_surfs, reflec):
    JM_dI = calc_jewel_throughput(n_surf, ref)
    print(
        f"Jewel mask with {n_surf} surfaces and reflectance of {ref:.3f} transmits %{JM_dI:.2f} of incoming light, which is {JM_dI / OM_dI:.3f} times greater than the opaque mask"
    )
