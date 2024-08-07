"""
Classes for working with different optical materials
"""

import numpy as np



class Material:
    """A class for working with optical materials"""

    def __init__(self, name: str):
        """An optical material from a given name


        Args:
            name (str): The common name or chemical composition of the material
                e.g. sio2, mgf2 etc.
        """
        self.name = name
        self.B, self.C = self._lookup_material(name)

    def refractive_index(self, wavelength: np.array):
        """
        Calculate the refractive index of the material at a given wavelength

        Parameters:
        -----------
        wavelength: np.array
            The wavelength in meters

        Returns:
        -----------
        n : np.array
            The refractive index of the material at each wavelength
        """
        wavelength = wavelength * 1e6  # convert to microns

        n_sqr = np.ones_like(wavelength)
        for b, c in zip(self.B, self.C):
            n_sqr += b * wavelength**2 / (wavelength**2 - c)
        return np.sqrt(n_sqr)

    @staticmethod
    def _lookup_material(glass: str):
        """
        Lookup the material in the refractiveindex.info database
        
        Very much copied/made OOD from opticstools library
        https://github.com/mikeireland/opticstools/blob/master/opticstools/opticstools.py

        Coefficents all assume wavelength is in units of microns
        """
        if glass == "sio2" or glass == "infrasil":
            B = np.array([0.696166300, 0.407942600, 0.897479400])
            C = np.array([4.67914826e-3, 1.35120631e-2, 97.9340025])
        elif glass == "bk7":
            B = np.array([1.03961212, 0.231792344, 1.01046945])
            C = np.array([6.00069867e-3, 2.00179144e-2, 1.03560653e2])
        elif glass == "nf2":
            B = np.array([1.39757037, 1.59201403e-1, 1.26865430])
            C = np.array([9.95906143e-3, 5.46931752e-2, 1.19248346e2])
        elif glass == "nsf11":
            B = np.array([1.73759695e00, 3.13747346e-01, 1.89878101e00])
            C = np.array([1.31887070e-02, 6.23068142e-02, 1.55236290e02])
        elif glass == "h-fk95n":
            B = np.array([0.973027043, 0.077363216, 0.687944956])
            C = np.array([0.004919458, 0.01820094, 159.44609])
        elif glass == "h-fk71":
            B = np.array([0.22226661, 0.880444178, 0.797848915])
            C = np.array([0.014065122, 0.004204574, 171.029818])
        elif glass == "ncaf2" or glass == "caf2":
            B = np.array([0.5675888, 0.4710914, 3.8484723])
            C = np.array([0.050263605, 0.1003909, 34.649040]) ** 2
        elif glass == "nsk16":
            B = np.array([1.34317774, 0.241144399, 0.994317969])
            C = np.array([0.00704687339, 0.0229005, 92.7508526]) ** 2
        elif glass == "nlasf9":
            B = np.array([2.00029547, 0.298926886, 1.80691843])
            C = np.array([0.0121426017, 0.0538736236, 156.530829]) ** 2
        elif glass == "mgf2":
            B = np.array([0.48755108, 0.39875031, 2.3120353])
            C = np.array([0.04338408, 0.09461442, 23.793604]) ** 2
        elif glass == "npk52a":
            B = np.array([1.02960700e00, 1.88050600e-01, 7.36488165e-01])
            C = np.array([5.16800155e-03, 1.66658798e-02, 1.38964129e02])
        elif glass == "psf67":
            B = np.array([1.97464225e00, 4.67095921e-01, 2.43154209e00])
            C = np.array([1.45772324e-02, 6.69790359e-02, 1.57444895e02])
        elif glass == "npk51":
            B = np.array([1.15610775e00, 1.53229344e-01, 7.85618966e-01])
            C = np.array([5.85597402e-03, 1.94072416e-02, 1.40537046e02])
        elif glass == "nfk51a":
            B = np.array([9.71247817e-01, 2.16901417e-01, 9.04651666e-01])
            C = np.array([4.72301995e-03, 1.53575612e-02, 1.68681330e02])
        elif (
            glass == "si"
        ):  # https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg
            B = np.array([10.6684293, 0.0030434748, 1.54133408])
            C = np.array([0.301516485, 1.13475115, 1104]) ** 2
        # elif (glass == 'zns'): #https://refractiveindex.info/?shelf=main&book=ZnS&page=Debenham
        #    B = np.array([7.393, 0.14383, 4430.99])
        #    C = np.array([0, 0.2421, 36.71])**2
        elif (
            glass == "znse"
        ):  # https://refractiveindex.info/?shelf=main&book=ZnSe&page=Connolly
            B = np.array([4.45813734, 0.467216334, 2.89566290])
            C = np.array([0.200859853, 0.391371166, 47.1362108]) ** 2
        elif glass == "sapphire" or glass == "al2o3":
            B = np.array([1.4313493, 0.65054713, 5.3414021])
            C = np.array([0.0726631, 0.1193242, 18.028251]) ** 2
        elif glass == "borofloat33":
            B = np.array([1.08161685e00, 5.89370752e-02, 7.68332358e-01])
            C = np.array([7.18522469e-03, 2.69899828e-02, 7.86152611e01])
        else:
            raise ValueError(f"Glass {glass} not found in database")
        return B, C

    @staticmethod
    def get_glass_name_list():
        """return the list of glass names available in the database"""

        return [
            "sio2",
            "bk7",
            "nf2",
            "nsf11",
            "ncaf2",
            "nsk16",
            "nlasf9",
            "mgf2",
            "npk52a",
            "psf67",
            "npk51",
            "nfk51a",
            "si",
            "znse",
            "sapphire",
            "borofloat33",
        ]
