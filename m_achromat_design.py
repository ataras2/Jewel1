"""
This script generates figure 6 of the paper. It shows the deviation of the achromat
as a function of wavelength for different materials and combinations of materials.
"""

from Telescopes import telescope_factory
from Materials import Material
from Jewels import AchromatSolver

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt


tscope = telescope_factory("LBT")
optim_wavel = 2.1e-6

# wavel_range = np.linspace(0.4e-6, 0.8e-6, 100)
print(tscope.get_filter_bands()[0, 0], tscope.get_filter_bands()[0, 1])
wavel_range = np.linspace(
    tscope.get_filter_bands()[0, 0], tscope.get_filter_bands()[-1, 1], 100
)

materials_list = ["sio2", "mgf2"]
devs = AchromatSolver.solve_over_combinations(
    tscope, materials_list, wavel_range, optim_wavel, 1
)

diff_limit = np.min(tscope.diffraction_limit.to(u.mas).value)


for mat, dev in zip(materials_list, devs):
    plt.plot(wavel_range * 1e6, dev.to(u.arcsec).value, label=f"{mat}")


materials_list = ["sio2", "caf2"]
devs = AchromatSolver.solve_over_combinations(
    tscope, materials_list, wavel_range, optim_wavel, 2
)
plt.plot(
    wavel_range * 1e6,
    devs[0].to(u.arcsec).value,
    label=f"{materials_list[0]} + {materials_list[1]}",
)

# save ylim
ylim = plt.gca().get_ylim()


diff_fraction = 0.25 / 1.22  # fraction of the diffraction limit that is acceptable
diff_offset = tscope.diffraction_limit.to(u.arcsec).value[1] * diff_fraction

plt.axvline(x=optim_wavel * 1e6, color="r", linestyle="--")
plt.axhline(
    y=tscope.on_sky_seperation.to(u.arcsec).value + diff_offset,
    color="k",
    linestyle="--",
)
plt.axhline(
    y=tscope.on_sky_seperation.to(u.arcsec).value - diff_offset,
    color="k",
    linestyle="--",
)

# plot filter bands as shades using axis bounds
for i, filt in enumerate(tscope.filters):
    kwargs = {}
    if i == 0:
        kwargs = {"label": "LBT filters"}
    plt.fill_between(
        [filt.low * 1e6, filt.high * 1e6],
        plt.gca().get_ylim()[0],
        plt.gca().get_ylim()[1],
        alpha=0.1,
        color="red",
        **kwargs,
    )


print(tscope.diffraction_limit.to(u.mas))

plt.ylim(ylim)

plt.xlabel("Wavelength ($\mu$m)")
plt.ylabel("Deviation (arcsec)")
plt.legend()

plt.show()
