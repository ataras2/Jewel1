
import itertools
import numpy as np
import astropy.units as u
import scipy.optimize as opt

from Materials import Material
from Telescopes import Telescope


class AchromatSolver:
    """This class works with telescope and material information to determine the
    parameters of the mask.
    """

    def __init__(
        self, tscope: Telescope, materials: list, wavel_range, optim_wavel
    ) -> None:
        """A solver to figure out the best combination of materials and wedge
            angles

        Args:
            tscope (Telescope): The telescope for which the achromat is
                designed
            materials (list): A list of materials to use in the achromat.
                Can be materials objects or strings
            wavel_range (np.ndarray): values at which we care about the
                chromatic dispersion
            optim_wavel (np.ndarray): the wavelength at which the deviation
                should be correct
        """
        self.tscope = tscope

        assert len(materials) != 0, "Must provide at least one material"

        if isinstance(materials[0], str):
            materials = [Material(m) for m in materials]

        self.materials = materials
        self.wavel_range = wavel_range
        self.optim_wavel = optim_wavel

        self.desired_deviation = (
            self.tscope.on_sky_seperation * self.tscope.magnification
        )

    def deviation_per_wedge(
        self, wedge_angles: np.ndarray, wavels: np.ndarray
    ):
        """Compute the deviation on the beam at each wavelength
        and for each wedge

        Args:
            wedge_angles (np.ndarray): The angles of each wedge
            wavels (np.ndarray): The wavelengths [m] for which to evaluate
                the deviation

        Returns:
            np.ndarray: the deviations as an array of shape
                (len(wedge_angles), len(wavels))
        """
        if not hasattr(wedge_angles, "unit"):
            wedge_angles *= u.arcmin

        dev = np.zeros([len(wedge_angles), len(wavels)]) * u.arcmin

        for i, (material, wedge_angle) in enumerate(
            zip(self.materials, wedge_angles)
        ):
            dev[i] = wedge_angle * (material.refractive_index(wavels) - 1)
        return dev

    def deviation(self, wedge_angles: np.ndarray, wavels: np.ndarray):
        """Calculate the deviation of the beam at a given wavelength

        Args:
            wedge_angles (np.ndarray): An array of angles for each of the
                wedges, in arcminutes or other astropy units
            wavels (np.ndarray): Wavelengths at which to calculate the
                deviation

        Returns:
            np.ndarray: The deviation of the beam at each wavelength
        """
        dev = self.deviation_per_wedge(wedge_angles, wavels)
        return dev.sum(axis=0)

    def solve_single_material(self):
        """Solve for the wedge angle of a single material

        Returns:
            np.ndarray: An array of wedge angles which, if only a
                single material is used, will deviate the beam the
                desired amount
        """
        wedge_angles = np.zeros(len(self.materials)) * u.arcmin
        for i, material in enumerate(self.materials):
            wedge_angles[i] = self.desired_deviation / (
                material.refractive_index(self.optim_wavel) - 1
            )
        return wedge_angles

    def endpoint_dispersion(self, wedge_angles: np.ndarray):
        """Calculate the goodness of the dispersion by using the
        absolute difference of the endpoints

        Args:
            wedge_angles (np.ndarray): An array of angles for each of the
                wedges

        Returns:
            float: The chromatic dispersion as the absolute difference of the
                endpoints of the waveband
        """
        dev = self.deviation(wedge_angles, self.wavel_range)
        return np.abs(dev[-1] - dev[0])

    def max_variation_dispersion(self, wedge_angles: np.ndarray):
        """The chromatic dispersion as the difference between the
        maximum and minimum deviation of the beam in the waveband

        Args:
            wedge_angles (np.ndarray): An array of angles for each of the
                wedges

        Returns:
            float: The chromatic dispersion as the difference between the
                maximum and minimum deviation of the beam in the waveband
        """
        dev = self.deviation(wedge_angles, self.wavel_range)
        return dev.max() - dev.min()

    def solve_achromat(
        self,
        init_guess: np.ndarray,
        metric_fun: callable,
        disp_progress: bool = False,
    ):
        """solves the optimization problem subject to the constraints that
        the deviation at the optim_wavel is correct

        Args:
            init_guess (np.ndarray): An array of initial guesses for the
                wedge angles
            metric_fun (callable): A function that takes in the wedge angles
                and returns a metric to optimize
            disp_progress (bool, optional): Whether to display the progress

        Returns:
            OptimizeResult : The result of the optimization. The optimal wedge
                angles are stored in .x
        """

        # constraint that the deviation at the optim_wavel is correct
        def deviation_constraint(wedge_ang):
            return (
                self.deviation(wedge_ang, np.array([self.optim_wavel]))
                - self.desired_deviation
            ).value

        x_opt = opt.minimize(
            metric_fun,
            init_guess,
            args=(),
            constraints={
                "type": "eq",
                "fun": deviation_constraint,
            },
            method="trust-constr",
            options={"disp": disp_progress},
        )

        return x_opt

    @staticmethod
    def solve_over_combinations(
        tscope: Telescope,
        materials_str: list[str],
        wavel_range: np.ndarray,
        optim_wavel: float,
        n_materials: int,
    ):
        """Given a list of many available materials, search over all
        combinations of n_materials to find the best achromat

        Args:
            tscope (Telescope): The telescope for which the achromat is
            materials_str (list[str]): A list of materials to possibly use in the achromat
            wavel_range (np.ndarray): values at which we care about the chromatic dispersion
            optim_wavel (float): the wavelength at which the deviation should be correct
            n_materials (int): the number of materials to use in the achromat
        """
        assert len(materials_str) >= n_materials

        deviations = []

        for material_subset in itertools.combinations(
            materials_str, n_materials
        ):
            solver = AchromatSolver(
                tscope, material_subset, wavel_range, optim_wavel
            )
            init_guess = solver.solve_single_material()
            x_opt = solver.solve_achromat(
                init_guess,
                solver.max_variation_dispersion,
            )

            on_sky_dispersion = (
                solver.max_variation_dispersion(x_opt.x * u.arcmin)
                / solver.tscope.magnification
            )

            deviations.append(
                solver.deviation(x_opt.x * u.arcmin, wavel_range)
                / solver.tscope.magnification
            )
            print(
                f"{material_subset} has optimal wedge angles: \
                    {x_opt.x*u.arcmin} with chromatic dispersion \
                    {on_sky_dispersion.to(u.mas):.2f}"
            )
        return deviations
