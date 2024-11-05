import numpy as np
from Hexagons import HexMath
import astropy.units as u
import Materials


def calc_throughput(
    n_surfs,
    n_interferograms,
    is_ar_coated,
    is_glued_stack,
    material,
    wavel,
    tscope,
    d_sub_ap,
    shim_width,
    n_apertures_per_interferogram,
):
    assert len(n_surfs) == n_interferograms

    if is_glued_stack:
        n_surfs = np.clip(
            n_surfs, 2, 0
        )  # if glued, all surfaces are glued together and only see 2 surfaces

    # inference from above params
    n_glass = Materials.Material(material).refractive_index(wavel)

    if is_ar_coated:
        fresnel_reflection = 0.7 / 100  # e.g. thorlabs B coating
    else:
        fresnel_reflection = (
            (1 - n_glass) / (1 + n_glass)
        ) ** 2  # this is the reflection at the glass-air interface

    pupil_diameter = tscope.pupil * u.m

    secondary_in_pupil_space = tscope.secondary / tscope.primary * pupil_diameter

    open_pupil_area = (
        np.pi * (pupil_diameter / 2) ** 2 - np.pi * (secondary_in_pupil_space / 2) ** 2
    )
    open_pupil_transmission = 1.0

    open_pupil_throughput = open_pupil_area * open_pupil_transmission

    # now consider a standard apterture mask, no shims since no edge effects
    sub_aperture = HexMath(incircle_r=pupil_diameter * d_sub_ap / 2)

    sub_aperture_area = sub_aperture.area
    mask_area = sub_aperture_area * n_apertures_per_interferogram
    mask_transmission = 1.0

    mask_throughput = mask_area * mask_transmission

    # Now Jewel mask, with shims and transmission loss
    sub_aperture = HexMath(incircle_r=(pupil_diameter * d_sub_ap - shim_width) / 2)

    sub_aperture_area = sub_aperture.area
    indiv_pattern_area = sub_aperture_area * n_apertures_per_interferogram

    jewel_mask_throughput = 0
    for n in n_surfs:
        indiv_pattern_throughput = (1 - fresnel_reflection) ** n

        jewel_mask_throughput += indiv_pattern_area * indiv_pattern_throughput

    return open_pupil_throughput, mask_throughput, jewel_mask_throughput
