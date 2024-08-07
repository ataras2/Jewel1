"""
This file contains the classes for the telescopes and filters.
"""

from typing import Sequence, Generator
import numpy as np
import astropy.units as u
import astropy.coordinates


class Filter:
    def __init__(self, center, width) -> None:
        self.center = center
        self.width = width
        self.low = self.center - self.width / 2
        self.high = self.center + self.width / 2

    def transmission(self, wavels: np.array):
        """
        The transmission of the filter at a given wavelength

        Args:
            wavels (np.array): The wavelengths in meters
        """
        pass


class BoxFilter(Filter):
    def __init__(self, center: float, width: float) -> None:
        super().__init__(center, width)

    def transmission(self, wavels):
        return (wavels > self.low) & (wavels < self.high)


class Telescope:
    """Contains data about the telescope size, pupil parameters and filters
    All sizes are diameters in meters
    """

    def __init__(
        self,
        primary: float,
        secondary: float,
        pupil: float,
        filter_holder: float,
        filters: Sequence,
        longest_baseline: float,
        on_sky_seperation: u.Quantity,
    ) -> None:
        self.primary = float(primary)
        self.secondary = float(secondary)
        self.pupil = float(pupil)
        self.filter_holder = float(filter_holder)
        self.filters = filters
        self.longest_baseline = float(longest_baseline)
        self.on_sky_seperation = on_sky_seperation

        # check filters are in ascending order
        for i in range(len(self.filters) - 1):
            assert self.filters[i].center < self.filters[i + 1].center

    @property
    def magnification(self):
        return self.primary / self.pupil

    @property
    def diffraction_limit(self):
        wavelength = np.array([filt.low for filt in self.filters])
        return astropy.coordinates.Angle(
            1.22 * wavelength / self.longest_baseline,
            unit=u.radian,
        )

    def filter_configs(self) -> Generator[Filter, None, None]:
        """an iterator that gives telescopes with only a single filter

        Returns:
            Telescope: A telescope object with a single filter
        """
        for filt in self.filters:
            yield Telescope(
                primary=self.primary,
                secondary=self.secondary,
                pupil=self.pupil,
                filter_holder=self.filter_holder,
                filters=[filt],
                longest_baseline=self.longest_baseline,
                on_sky_seperation=self.on_sky_seperation,
            )

    def get_filter_bands(self):
        """Get the wavelengths where the filters transmit light

        Returns:
            np.array: with shape 2 x len(self.filters) that containts the min and max wavelengths
        """
        return np.array([[filt.low, filt.high] for filt in self.filters])

    def wavel_range_overall(self):
        """
        Compute the min and max wavelengths that the telescope can observe
        """
        return np.array([self.filters[0].low, self.filters[-1].high])


def telescope_factory(name: str) -> Telescope:
    """Uses a lookup of know telescopes to create a Telescope object

    Args:
        name (str): A valid name of a telescope. Case insensitive.

    Returns:
        Telescope: A telescope object that has been populated
    """

    if name.lower() == "dct":
        # filter_centers = np.array([, 658, 808, 880, 1150, 1570]) * 1e-9
        # bandwidths = np.array([40] * 4 + [50] * 2) * 1e-9
        filter_centers = np.array([592, 754, 820, 870]) * 1e-9
        bandwidths = np.array([39, 36, 37, 40]) * 1e-9
        scope = Telescope(
            primary=4.28,
            secondary=1.4,
            pupil=8.5e-3,
            filter_holder=25.4e-3,
            filters=[
                BoxFilter(center, width)
                for center, width in zip(filter_centers, bandwidths)
            ],
            longest_baseline=4.28,
            on_sky_seperation=astropy.coordinates.Angle('2.3"'),
        )
    elif name.lower() == "subaru_vis":
        filter_centers = np.array([600, 650, 700, 750]) * 1e-9
        bandwidths = np.array([50] * 4) * 1e-9
        scope = Telescope(
            primary=8.3,
            secondary=1.4,
            pupil=7.03e-3,
            filter_holder=25.4e-3,
            filters=[
                BoxFilter(center, width)
                for center, width in zip(filter_centers, bandwidths)
            ],
            longest_baseline=8.3,
            on_sky_seperation=astropy.coordinates.Angle(1.5, unit="arcsec"),
        )
    elif name.lower() == "subaru_nir":
        filter_centers = np.array([1150, 1570]) * 1e-9
        bandwidths = np.array([50] * 2) * 1e-9
        scope = Telescope(
            primary=8.3,
            secondary=1.4,
            pupil=17.92e-3,
            filter_holder=25.4e-3,
            filters=[
                BoxFilter(center, width)
                for center, width in zip(filter_centers, bandwidths)
            ],
            longest_baseline=8.3,
            on_sky_seperation=astropy.coordinates.Angle(1.5, unit="arcsec"),
        )
    elif name.lower() == "lbt":
        filter_centers = np.array([1.655, 2.16, 2.925]) * 1e-6
        bandwidths = np.array([0.31, 0.32, 0.11]) * 1e-6
        scope = Telescope(
            primary=23,
            secondary=0.9,
            pupil=7.49e-3,
            filter_holder=25.4e-3,
            filters=[
                BoxFilter(center, width)
                for center, width in zip(filter_centers, bandwidths)
            ],
            longest_baseline=23,
            on_sky_seperation=astropy.coordinates.Angle(5, unit="arcsec"),
        )
    else:
        raise ValueError(f"Unknown telescope name: {name}")

    return scope
