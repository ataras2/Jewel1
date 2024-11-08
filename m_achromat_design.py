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

np.random.seed(0)

tscope = telescope_factory("LBT")
optim_wavel = 2.1e-6

# wavel_range = np.linspace(0.4e-6, 0.8e-6, 100)
print(tscope.get_filter_bands()[0, 0], tscope.get_filter_bands()[0, 1])
wavel_range = np.linspace(
    tscope.get_filter_bands()[0, 0], tscope.get_filter_bands()[-1, 1], 100
)

materials_list = ["sio2", "mgf2"]
devs, _ = AchromatSolver.solve_over_combinations(
    tscope, materials_list, wavel_range, optim_wavel, 1
)

diff_limit = np.min(tscope.diffraction_limit.to(u.mas).value)


for mat, dev in zip(materials_list, devs):
    plt.plot(wavel_range * 1e6, dev.to(u.arcsec).value, label=f"{mat}")


materials_list = ["sio2", "caf2"]
devs, _ = AchromatSolver.solve_over_combinations(
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


# now consider non-ideal case
def add_diffraction_lines(deviations, diff_offset, use_central_band=False):
    # for the outer filter bands, add diffraction horizontal lines, but only for the width of the band
    for i, filt in enumerate(tscope.filters):
        if i == 1 and not use_central_band:
            continue
        wavel_index = np.argmin(np.abs(wavel_range - filt.center))
        plt.hlines(
            y=deviations[wavel_index] + diff_offset,
            xmin=filt.low * 1e6,
            xmax=filt.high * 1e6,
            color="k",
            linestyle="--",
        )
        plt.hlines(
            y=deviations[wavel_index] - diff_offset,
            xmin=filt.low * 1e6,
            xmax=filt.high * 1e6,
            color="k",
            linestyle="--",
        )


materials_list = ["sio2", "mgf2"]
devs, wedge_angles = AchromatSolver.solve_over_combinations(
    tscope, materials_list, wavel_range, optim_wavel, 1
)

n_instances = 100

wedge_angle_error_sigma = 5 * u.arcmin
# distribution = lambda x, size: np.random.normal(0, x.value, size=size) * x.unit
distribution = lambda x, size: np.random.uniform(-x.value, x.value, size=size) * x.unit

plt.figure()

# get the colours that matplotlib normally uses for plotting
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for i, (mat, dev, wedge_ang) in enumerate(zip(materials_list, devs, wedge_angles)):
    solver = AchromatSolver(tscope, [mat], wavel_range, optim_wavel)

    deviations = []
    for _ in range(n_instances):
        wedge_ang_instance = wedge_ang + distribution(wedge_angle_error_sigma, size=1)
        raw_deviations = (
            solver.deviation(wedge_ang_instance, wavel_range)
            / solver.tscope.magnification
        )

        # but really we want to remove the deviation at the design wavelength
        # so we can see the effect of the wedge angle
        raw_deviations -= raw_deviations[np.argmin(np.abs(wavel_range - optim_wavel))]
        deviations.append(raw_deviations)

    mean_devs = np.mean([dev.to(u.arcsec).value for dev in deviations], axis=0)
    std_devs = np.std([dev.to(u.arcsec).value for dev in deviations], axis=0)
    plt.plot(wavel_range * 1e6, mean_devs, color=colors[i])
    plt.fill_between(
        wavel_range * 1e6,
        mean_devs - 3 * std_devs,
        mean_devs + 3 * std_devs,
        color=colors[i],
        alpha=0.3,
    )

    add_diffraction_lines(mean_devs, diff_offset, use_central_band=False)

# now consider the achromat
materials_list = ["sio2", "caf2"]
devs, wedge_angles = AchromatSolver.solve_over_combinations(
    tscope, materials_list, wavel_range, optim_wavel, 2
)
wedge_angles = wedge_angles[0]  # only one combination was tested

# simulate the errors
solver = AchromatSolver(tscope, materials_list, wavel_range, optim_wavel)
deviations = []
for _ in range(n_instances):
    wedge_ang_instance = wedge_angles + distribution(wedge_angle_error_sigma, size=2)
    raw_deviations = (
        solver.deviation(wedge_ang_instance, wavel_range) / solver.tscope.magnification
    )

    # but really we want to remove the deviation at the design wavelength
    # so we can see the effect of the wedge angle
    raw_deviations -= raw_deviations[np.argmin(np.abs(wavel_range - optim_wavel))]
    deviations.append(raw_deviations)

mean_devs = np.mean([dev.to(u.arcsec).value for dev in deviations], axis=0)
std_devs = np.std([dev.to(u.arcsec).value for dev in deviations], axis=0)

plt.plot(wavel_range * 1e6, mean_devs, color=colors[2], label="")
plt.fill_between(
    wavel_range * 1e6,
    mean_devs - 3 * std_devs,
    mean_devs + 3 * std_devs,
    color=colors[2],
    alpha=0.3,
)


add_diffraction_lines(mean_devs, diff_offset, use_central_band=True)


plt.axvline(x=optim_wavel * 1e6, color="r", linestyle="--")

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


plt.xlabel("Wavelength ($\mu$m)")
plt.ylabel("Deviation error at design deviation (arcsec)")
plt.legend()
plt.ylim([-0.2, 0.1])

plt.show()
