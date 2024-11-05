import numpy as np
import astropy.units as u

from throughput import calc_throughput

import itertools

from Telescopes import telescope_factory
import Materials

# parameters read from the table in appendix:
n_interferograms = 4
n_apertures_per_interferogram = 4
d_sub_ap = 0.196

# application
tscope = telescope_factory("subaru_vis")
wavel = tscope.filters[0].center

# parameters for how much money you use on manufacturing:
shim_width = 140 * u.um
material = "mgf2"
n_surfs = (
    np.array([0, 1, 1, 2]) * 2
)  # an array of how many surfaces each interferogram sees


# start calculation here
for is_ar, is_glue in itertools.product([True, False], repeat=2):

    open_pupil_throughput, mask_throughput, jewel_mask_throughput = calc_throughput(
        n_surfs,
        n_interferograms,
        is_ar,
        is_glue,
        material,
        wavel,
        tscope,
        d_sub_ap,
        shim_width,
        n_apertures_per_interferogram,
    )

    print(f"Considering case: AR coated: {is_ar}, Glued stack: {is_glue}")
    print(
        f"AM: Percentage of open pupil throughput: {100*mask_throughput / open_pupil_throughput:.3f}%"
    )

    print(
        f"JM: Percentage of open pupil throughput: {100*jewel_mask_throughput / open_pupil_throughput:.3f}%"
    )
    print(
        f"Jewel mask is a factor {(jewel_mask_throughput / mask_throughput):.2f} times more efficient than conventional aperture masking"
    )


# Now repeat for a more complex case, e.g. the pattern in fig 11
# parameters read from the table in appendix:
n_interferograms = 7
n_apertures_per_interferogram = 6
d_sub_ap = 0.125

# application
tscope = telescope_factory("subaru_vis")
wavel = tscope.filters[0].center

# parameters for how much money you use on manufacturing:
shim_width = 20 * u.um
is_ar_coated = False
is_glued_stack = False
material = "mgf2"
n_surfs = (
    np.array([1, 1, 1, 2, 2, 2, 3]) * 2
)  # an array of how many surfaces each interferogram sees

# start calculation here
for is_ar, is_glue in itertools.product([True, False], repeat=2):

    open_pupil_throughput, mask_throughput, jewel_mask_throughput = calc_throughput(
        n_surfs,
        n_interferograms,
        is_ar,
        is_glue,
        material,
        wavel,
        tscope,
        d_sub_ap,
        shim_width,
        n_apertures_per_interferogram,
    )

    print(f"Considering case: AR coated: {is_ar}, Glued stack: {is_glue}")
    print(
        f"Percentage of open pupil throughput: {100*mask_throughput / open_pupil_throughput:.3f}%"
    )

    print(
        f"Percentage of open pupil throughput: {100*jewel_mask_throughput / open_pupil_throughput:.3f}%"
    )
    print(
        f"Jewel mask is a factor {(jewel_mask_throughput / mask_throughput):.2f} times more efficient than conventional aperture masking"
    )
