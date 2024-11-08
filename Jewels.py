
import itertools
import numpy as np
import astropy.units as u
import scipy.optimize as opt

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import dLux as dl
import dLux.utils as dlu

import jax.numpy as jnp
from jax import Array, vmap
import equinox as eqx

from Materials import Material
from Telescopes import Telescope
from Idl_file_reader import IDLFileReader


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
        optimal_solutions = []

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
            optimal_solutions.append(x_opt.x * u.arcmin)
            print(
                f"{material_subset} has optimal wedge angles: \
                    {x_opt.x*u.arcmin} with chromatic dispersion \
                    {on_sky_dispersion.to(u.mas):.2f}"
            )
        return deviations, optimal_solutions

class MachiningSequenceSolver:
    """
        Class to determine the optimal machining sequence for a given Jewel mask.

        MachiningSequenceSolver knows:
            - The number of wedges used in the Jewel mask
            - The number of tiling patterns in the Jewel mask
            - The location of hexagonal segments in each Jewel mask tiling pattern
            - The size of the hexagonal segments in the Jewel mask

        MachiningSequenceSolver can calculate the machining-optimised configuration
        of the Jewel mask. 

    """  
    def __init__(self, n_patterns: int, n_wedges: int, seg_centers: np.array, hex_size: float) -> None:
        """
            Parameters:
            ----------
            n_patterns: int
                Number of unique hexagonal patterns/tilings within Jewel mask
            n_wedges: int
                Number of wedges used in the Jewel mask
            seg_centers: np.array
                Has dimensions (n_patterns, n_segments, 2), containing the x,y coordinates
                of the centers of the hexagons per pattern, where n_segments represents
                the number of hexagonal segments in each pattern. Units in meters.
            hex_size: float
                Max radius to vertices from center of hexagonal segment, meters.

        """
        assert seg_centers.shape[-1] == 2, "Provided shape centers are not two-dimensional"

        # Priv vars
        self._n_patterns = n_patterns
        self._n_wedges = n_wedges
        self._seg_centers = seg_centers
        self._hex_size = hex_size
        self._machining_sequence = self.calc_machining_sequence()

    @property
    def machining_sequence(self):
        """
            Returns:
            --------
            list of arrays:
                Each element in the list corresponds to an array of indices representing the 
                tiling patterns that are machined onto wedge j. The length of the list is equal to the
                number of wedges.
        """
        
        return self._machining_sequence
        
    @staticmethod
    def count_neighbour_cuts(hex_centers: np.array, hex_size: float):
        """
            count how many neighbours have been cut too (the higher this number is, the
            less total cuts will be made)
            Parameters:
            ----------
            hex_centers: np.array
                2D array containing the [x,y] coordinates of the centers of the hexagons
                to be machined. Units in meters.
            hex_size: float
                Size of hexagon in meters

        """
        n_neighbours = 0
        for point in hex_centers:
            dists_to_others = (
                np.linalg.norm(point - hex_centers, axis=1) / hex_size
            )
            n_neighbours += (dists_to_others < 2).sum() - 1

        return n_neighbours / 2
    
    def calc_machining_sequence(self):
        """
            Calculates the optimal machining configuration (i.e. which patterns are machined on which wedge(s))
            by combining patterns that minimise the perimeter of the machined pattern (all wedges considered)

            Returns:
            -----------
            wedge_list: list of np.arrays
                Element j in the list corresponds to an array of pattern IDs representing the
                tiling patterns that are machined onto wedge j. The length of the list is equal to the
                number of wedges.
                
        """

        # Reproduce every possible machining sequence per tiling pattern
        pattern_seq_permu = itertools.permutations(range(2**self._n_wedges), self._n_patterns) #2^N possible sequences mapped to n_patterns
        stored_bin_perm = []
        total_neighbours_comb = np.array([]) #store total number of neighbours for each permu
        for i_pat_seq, pat_seq in enumerate(pattern_seq_permu):
            # eg pat_seq = (0,3,2,1)-> pattern 0 has sequence 00, pattern 1 has sequence 11, pattern 2 has sequence 10 etc..
            # create binary equivalent of the above (eg (0,3,2,1) = (00,11,10,01))
            pat_seq_bin = ()
            for num in pat_seq:
                pat_seq_bin = pat_seq_bin + (bin(num)[2:].zfill(self._n_wedges),)

            stored_bin_perm.append(pat_seq_bin)
            total_neighbours = 0
            for wedge in range(self._n_wedges):
                machined_centers = np.array([]) #Array containing centers that are machined on current wedge
                for pat in range(self._n_patterns):
                    seq = pat_seq_bin[pat] # bin sequence for current pattern
                    hole_bool = seq[wedge] #bool corresponding to if the current pattern on the current wedge
                                            # is machined (1 = yes)

                    if hole_bool == "1":
                        if machined_centers.size == 0:
                            machined_centers = self._seg_centers[pat,:]
                        else:
                            machined_centers = np.append(machined_centers, self._seg_centers[pat,:], axis = 0)

                total_neighbours += self.count_neighbour_cuts(machined_centers, self._hex_size)
            
            total_neighbours_comb = np.append(total_neighbours_comb, total_neighbours)
                        
        max_neighbours_idx = np.argmax(total_neighbours_comb) # Pick first max
        optimal_pat = stored_bin_perm[max_neighbours_idx]
        
        # Store in list of arrays. Entry i contains the pattern idxs for tiling patterns machined on wedge i
        wedge_list = []
        for wedge_i in range(self._n_wedges):
            idxs = np.array([])
            for pat_i, char_bool in enumerate(optimal_pat):
                thru  = char_bool[wedge_i]
                if thru == "1":
                    idxs = np.append(idxs, int(pat_i))

            wedge_list.append(idxs.astype(int))

        return wedge_list


class MaskPattern:
    """
        MaskPattern knows the general geometrical features describing a Jewel mask, including:
            - The number of hexagonal patterns and the number of hexagons in each
            - The relative position of the hexagonal segments on the mask
            - The primary (and optional secondary) diameter of the mask
            - The underlying unitless dimension of the grid describing the pattern
            - The physical grid size that the pattern will be tesselated onto
            - The shim width (i.e. spacing) between hexagons
            - Any transformations to the __entire__ original pattern (e.g. rotation, translation)

        This information can be given directly or via the IDL .dat file.

        MaskPattern can calculate:
            - The cartesian coordinates of the hexagonal centers on the mask (with any transformations
             to the whole pattern applied)
            - The required number of wedges to achieve the specfied 
                number of patterns deflected onto the detector
            - The maximum radius of the hexagonal segments (from center to vertex) in 
                order to achieve desired separation given physical grid size
            - The machiniability-optimised with which each pattern should be machined 
            from the wedge(s).
    """

    def __init__(
        self,
        n_pat: int,
        n_seg: int,
        hex_idxs: np.array,
        idim: int, 
        primary_diam: float,
        grid_size: float | tuple,
        shim_width: float, 
        jdim: int = None,
        secondary_diam: float = 0.0,
        tf: jnp.array = None,
        manual_machining_seq: list[jnp.array] = None,
    ) -> None:
        """
            Parameters:
            ----------
            n_pat: int
                Number of unique hexagonal patterns/tilings within Jewel mask
            n_seg: int
                Number of hexagonal segments within each unique tiling pattern
            hex_idxs: np.array
                Center idx's for hexagon on triganular grid. Shape (n_tilings, n_seg), where
                row i contains the center idx's for the ith tiling pattern. Index positions are 
                read on the tringular grid from bottom to top, left to right.
            idim: int
                Underlying horizontal dimension of grid (unitless) where hex_idxs are defined
                upon.
            primary_diam: float
                Primary diameter of Jewel mask in meters
            grid_size: float or 2D tuple of floats
                Dimension of the underlying rectangular grid across pupil, in meters.
                If single value, assumed grid is square. If 2D tuple, then grid is rectangular,
                with first entry horiztonal dimension.
                In general, will be one unit less than primary_diam? TODO check
            shim_width: float
                Shim bridge width between adjacent hexagons. Units in meters.
            jdim: int = None, optional
                Underlying vertical dimension of grid (unitless) where hex_idxs are defined. If None
                then assumed to be equal to idim.
            secondary_diam: float, optional
                Secondary diameter of Jewel mask in meters
            tf: jnp.array = None, optional
                (3,) Array consisting of [x translation, y translation, rotation] to apply to
                _entire_ jewel pattern. Units in meters, meters and radians respectively.
            manual_machining_seq: list[jnp.array] = None, optional
                List of arrays, each array corresponds to an array of indices representing the
                tiling patterns that are machined onto wedge j. The length of the list is equal to the
                number of wedges. 

                If left as None, machining sequence is calculated automatically using MachiningSequenceSolver

        """
        assert hex_idxs.shape == (n_pat, n_seg), """Format of given hexagon indices is incorrect.
                                                    Expecting: ({},{})""".format(n_pat, n_seg)
        

        assert isinstance(grid_size, float | tuple), "Grid size must be a single float or 2D tuple of floats."
        if isinstance(grid_size, tuple):
            assert len(grid_size) == 2, "Grid size tuple must have two elements."

        #Private vars
        self._n_pat = n_pat
        self._n_seg = n_seg
        self._hex_idxs  = hex_idxs
        self._idim = idim
        self._jdim = jdim
        self._primary_diam = primary_diam
        self._secondary_diam = secondary_diam   
        self._grid_size = grid_size
        self._shim_width = shim_width    
        self._rmax = self.get_rmax() 
        self._tf = tf

        self._cart_hex_centers = self.calc_cart_hex_centers # calc once and store

        # If a machining pattern is not given, calculate the optimal machining sequence
        if manual_machining_seq is not None:
            assert len(manual_machining_seq) == self.n_wedges, "Number of wedges does not match number of machining sequences given."
            self._machining_sequence = manual_machining_seq
        else:
            self._machining_sequence = MachiningSequenceSolver(self._n_pat, self.n_wedges, self.cart_hex_centers, self._rmax).machining_sequence


    # Alternative constructor using IDL file-derived info
    @classmethod
    def from_file(cls, fname: str, 
                  solution_idx: int, 
                  primary_diam: float, 
                  grid_size: int, 
                  shim_width: float, 
                  secondary_diam: float = 0.0, 
                  tf: jnp.array = None, 
                  manual_machining_seq: list[jnp.array] = None,
                  ):
        """
            Parameters:
            ----------
            fname: str
                String to filepath of IDL file
            solution_idx: int
                Solution index to pick which hexagonal pattern array from IDL file to return centers of.

            Rest of the parameters are defined in MaskPattern constructor

            Returns:
            --------
            MaskPattern object
        """
        # Load relevant pattern info from IDL file
        IDLfile = IDLFileReader(fname)
        n_pat, n_seg, idim, jdim = IDLfile.pattern_info
        hex_idxs = IDLfile.get_hex_centers_idxs(solution_idx)  

        return cls(n_pat=n_pat, n_seg=n_seg, hex_idxs = hex_idxs, idim = idim, jdim = jdim, primary_diam = primary_diam, 
                   secondary_diam= secondary_diam, grid_size = grid_size, shim_width = shim_width, tf = tf,
                   manual_machining_seq = manual_machining_seq)

    @property
    def machining_sequence(self):
        """
            Returns:
            --------
            list of arrays:
                Each element in the list corresponds to an array of indices representing the 
                tiling patterns that are machined onto wedge j. The length of the list is equal to the
                number of wedges.
        """
        return self._machining_sequence

    @property
    def n_tilings(self):
        """
            Returns:
            --------
            int: 
                Number of unique hexagonal patterns/tilings within Jewel mask
        """
        return self._n_pat
    
    @property
    def seg_rotation(self):
        """
            Returns:
            --------
            float: 
                Rotation in radians of each segment. Default is pi/6 unless
                pattern transform is given.
                
        """
        to_return = np.pi/6

        if self._tf is not None:
            to_return = np.pi/6 - self._tf[2]

        return to_return

    @property
    def primary_diam(self):
        """
            Returns:
            --------
            float: 
                Primary diameter of Jewel mask
        """
        return self._primary_diam
    
    @property
    def rmax(self):
        """
            Returns:
            --------
            float: 
                Max radius to vertices from center of hexagonal segment, meters.
                Assuming a set grid separation, rmax is calculated from a desired
                shim width.
        """
        return self._rmax
    
    @property
    def n_segments(self):
        """
            Returns:
            --------
            int: 
                Number of hexagonal segments within each unique tiling pattern
        """
        return self._n_seg
    
    @property
    def n_wedges(self):
        n_bits = len(bin(self._n_pat)[2:])  # Number of bits used for number of patterns bin representation
        bit_thresh = 2**(n_bits-1)          # MSB high, rest 0 - same size as n_bits
        diff = self._n_pat - bit_thresh
        _n_wedges = (n_bits-1) if diff == 0 else n_bits  # If exact number of wedges meets # of patterns required (N wedges yields 2^N patterns), then set this
                                                        # else move one power up the bin representation
        return _n_wedges
    
    @property
    def cart_hex_centers(self):
        """
            Returns:
            --------
            np.array:
                Has dimension (_n_pat, _n_seg, 2), containing the x,y coordinates
                of the centers of the hexagons per pattern. Units in meters.
        """
        return self._cart_hex_centers
    
    @property
    def shim_bridge_width(self):
        """
            Returns:
            --------
            float:
                Width of the shim bridge in meters
        """
        if isinstance(self._grid_size, float):
            grid_sep = self._grid_size/self._idim

        else: 
            grid_sep = self._grid_size[0]/self._idim

        width = (grid_sep * self._hscale) - (np.sqrt(3) * self._rmax)

        return width
    
    def get_rmax(self):
        """
            Returns:
            --------
            float:
                Max radius to vertices from center of hexagonal segment, meters.
                Assuming a set grid separation, rmax is calculated from a desired
                shim width.
        """
        if isinstance(self._grid_size, float):
            grid_sep = self._grid_size/self._idim

        else: 
            grid_sep = self._grid_size[0]/self._idim

        rmax = (grid_sep - self._shim_width)/np.sqrt(3)

        return rmax
    
    # def get_hscale(self):
    #     """
    #         Returns:
    #         --------
    #         float:
    #             Scale factor applied to hexagonal grid determining the spacing
    #             between hexagonal segements
    #     """
    #     if isinstance(self._grid_size, float):
    #         grid_sep = self._grid_size/self._idim

    #     else: 
    #         grid_sep = self._grid_size[0]/self._idim

    #     hscale  = (self._shim_width + np.sqrt(3) * self._rmax) / grid_sep

    #     return hscale
    def set_cart_hex_centers(self, cart_centers):
        """
            Parameters:
            -----------
            cart_centers: np.array
                (n_pat, n_seg, 2) size array containing the [x,y] coordinates of the centers of the hexagons
                to be machined. Units in meters.
        """
        self._cart_hex_centers = cart_centers


    @property
    def calc_cart_hex_centers(self):
        """
            Returns:
            --------
            np.array:
                Has dimension (_n_pat, _n_seg, 2), containing the x,y coordinates
                of the centers of the hexagons per pattern. Units in meters.
        """
        if isinstance(self._grid_size, float):
            # Square-grid case
            grid_sep = self._grid_size/self._idim

            hix = np.arange(-(self._grid_size-grid_sep)/2, (self._grid_size-grid_sep)/2+grid_sep, grid_sep)
            hxx,hyy = np.meshgrid(hix,hix)

            assert len(hix)**2 > (np.max(self._hex_idxs)), """Grid indices will not fit under current underlying grid specifications.
                                                            Getting {}x{} and maximum idx is {}""".format(len(hix), len(hix), np.max(self._hex_idxs))

        else:
            # Rectangular-grid case
            grid_sep = self._grid_size[0]/self._idim  # should be identical in both directions? TODO

            hix = np.arange(-(self._grid_size[0]-grid_sep)/2, (self._grid_size[0]-grid_sep)/2+grid_sep, grid_sep)
            hiy = np.arange(-(self._grid_size[1]-grid_sep)/2, (self._grid_size[1]-grid_sep)/2+grid_sep, grid_sep)
            hxx,hyy = np.meshgrid(hix,hiy)

            assert len(hix)*len(hiy) > (np.max(self._hex_idxs)), """
                                    Grid indices will not fit under current underlying grid specifications.
                                    Getting {}x{} and maximum idx is {}""".format(len(hix), len(hiy), np.max(self._hex_idxs))

        # Shear grid 
        hxx = (hxx - hyy / 2.0)        
        hyy = (hyy * np.sqrt(3.0) / 2.0) 

        # # Centering : this does not always center nicely regardless - apply tf to center pattern manually
        # hxx = hxx - (scale*grid_sep) / 4.0
        # hyy = hyy + (scale*grid_sep) / 4 * 0.57735

        hxxv = np.reshape(hxx, -1)
        hyyv = np.reshape(hyy, -1) 
                               
        centres = []
        for pattern in self._hex_idxs:
            for coord in pattern:
                centres.append((hxxv[coord], hyyv[coord]))
        
        centres = np.asarray(centres)

        # Apply transformation if given
        if self._tf is not None:
            assert self._tf.shape == (3,), "Transformation array is not of correct shape. Expecting (3,)"
            x_trans = self._tf[0]
            y_trans = self._tf[1]
            rot_mat = rotation_matrix(self._tf[2]) 
            for i, cen in enumerate(centres):
                cen += jnp.array([x_trans, y_trans])
                cen  = jnp.matmul(cen, rot_mat)

                centres[i] = cen

        # Convert to array of dimension (n_pat, n_seg, 2)
        centres = np.reshape(centres, (self._n_pat,self._n_seg,2)) 

        return centres

    
    def display_pattern(self, plt_show = True):
        """
            Returns:
            --------
            None:
                Displays the Jewel mask pattern
        """
        colour_vect = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

        plt.figure(figsize=(5,5))
        ax = plt.gca()  
        lgnd_patches = []
        for i, pattern in enumerate(self.cart_hex_centers):
            patch = 0
            for seg_coord in pattern:
                patch = patches.RegularPolygon((seg_coord[0], seg_coord[1]), 6, self._rmax, color=colour_vect[i], 
                                               label = str(i), orientation=-self.seg_rotation + np.pi/6, fill=True)
                ax.add_artist(patch)
            lgnd_patches.append(patch)

        primary_m = plt.Circle((0.0, 0.0), self._primary_diam / 2.0, color="black", fill=False)
        secondary_m = plt.Circle((0.0, 0.0), self._secondary_diam / 2.0, color="black", fill=False)
        ax.add_artist(primary_m)
        ax.add_artist(secondary_m)
        cen = plt.scatter(0,0, s=1)
        ax.add_artist(cen)
        ax.set(xlim=(-self._primary_diam/2*1.1, self._primary_diam/2*1.1), ylim=(-self._primary_diam/2*1.1, self._primary_diam/2*1.1))
        
        plt.title("Jewel Mask Pattern")
        plt.legend(handles = lgnd_patches)
        plt.axis('off')

        if plt_show:
            plt.show()

    
    
class JewelWedge(dl.layers.BasisOptic):
    """
        A JewelWedge is-a dLux BasisOptic layer, which applies tip & tilt aberrations (achieveing the desired deviation
        of light passing through the wedge) where hexagonal segments are __not__ machined, and can control the tranmission 
        of the wavefont (according to any transmittance specifications for the glass used). Class accounts for chromatic
        dispersion in apply() method.

        A JewelWedge knows:
            - Its wedge angle
            - The material it is made of (knows-a Material())
            - The slope orientation of the wedge (relative to static hex pattern)
            - Its physical diameter (primary diameter) and the number of pixels across the diameter
            - The size of each hexagonal segement machined onto it
            - The 2D coordinates for each pixel center in JewelWedge
            - The location of hexagonal segments machined onto it
            - The tip/tilt zernike coefficients describing the physical deviation of light applied by the wedge
            - The transmittance value for the glass wedge
            - Any transformations to the location of each machined hex segment (representing machining errs)

    """
    zernike_idxs = jnp.array([2,3]) # Tip/Tilt Zernike (noll) indices
    rot = jnp.pi/6                  # Default rotation of hexagonal segments (longest axis aligns with y-axis)
    n_sides = 6                     # Number of sides on each hexagonal segment

    _machined_centers_trans: jnp.array
    _wedge_transmission: jnp.array
    _wedge_angle: float
    _material: Material

    def __init__(
            self: dl.layers.optical_layers.OpticalLayer,
            wedge_angle: float,
            slope_orientation: float,
            material: Material,
            prim_diam: float, 
            rmax: float, 
            n_pix: int,
            pixel_coords: jnp.array,
            hex_centers: jnp.array,
            machined_centers_idxs: jnp.array,
            # coefficients: jnp.array,
            glass_trans: float = None,
            machined_centers_tf: jnp.array = None,
            as_phase: bool =False, 
            normalise: bool=False,
        ) -> None:
        """
            Parameters:
            ----------
            wedge_angle: float
                Angle of the wedge in degrees (> 0)
            slope_orientation: float
                Orientation of the wedge slope relative to the static hex pattern (radians).
                Rotation starts from positive x-axis and moves CCW with initial slope orientation 
                horizontal and highest point on the wedge furthest to the right.
            material: Material
                Material object representing the glass wedge
            prim_diam: float
                Primary diameter of Jewel mask in meters
            rmax: float
                Max radius to vertices from center of hexagonal segment, meters.
            n_pix: int
                Number of pixels spanning vertically/horizontally over the primary mask diameter
            pixel_coords: jnp.array
                2D coordinates for each pixel center in JewelWedge (cartesian or polar)
            hex_centers: jnp.array
                3D array describing the cartesian [x,y] coordinates of __every__ hexagonal segments'
                center in every tiling pattern in the entire Jewel mask. Has dimension (n_pat, n_seg, 2)
                where n_pat = the # of tiling patterns, n_seg = the # of segments in each tiling pattern
                on the Jewel mask.
            machined_centers_idxs: jnp.array
                1D array with each entry containing the pattern idx that is machined onto this JewelWedge
            # coefficients: jnp.array
            #     Tip/Tilt Zernike coefficients describing the deviation of light as it passes through the wedge
            glass_trans: float = None
                Transmittance value for each glass wedge, range [0,1].
            machined_centers_tf: jnp.array = None
                Array of transformations to apply to each hexagonal segments. Length should be equal to
                machined_centers. Size (n,3) for n machined hexagonal segments containing a [x translation,
                y translation, rotation] per segment.
            as_phase : bool = False
                Whether to apply the basis as a phase or OPD. If True the basis is
                applied as a phase, else it is applied as an OPD.
            normalise : bool = False
                Whether to normalise the wavefront after passing through the JewelWedge optic.
        """
        assert hex_centers.shape[-1] == 2, "Hexagonal centers dimensions are in correct. Expecting (n_patterns, n_segments,2)"
        # assert len(coefficients) == 2, "Expecting 2 Zernike coefficients for tip/tilt"

        if glass_trans is not None:
            assert glass_trans >= 0 and glass_trans <= 1, "Glass transmittance must be in range [0,1]"

        assert wedge_angle > 0, "Wedge angle must be positive"

        self._wedge_angle = wedge_angle
        self._material = material
        coefficients = self.calc_base_coeffs(slope_orientation, prim_diam)

        # Calculate the transmission arrays for the machined hexagonal segments only. This will be used
        # to determine a wedge basis used for the application of aberrations. Accounts for any unideal
        # translations/rotations in the position of the machined segments
        _machined_centers = hex_centers[machined_centers_idxs,:]
        _machined_centers = np.reshape(_machined_centers, (-1,2)) #2D [x,y] cartesian coordinate for each center
        _machined_centers_tf = self.machined_centers_tfs(machined_centers_tf, _machined_centers)
        self._machined_centers_trans = self.machined_centers_transmission(rmax, _machined_centers_tf, pixel_coords, prim_diam, n_pix)

        _wedge_basis = self.wedge_basis(pixel_coords, prim_diam, self._machined_centers_trans)
        self._wedge_transmission = self.wedge_transmission(pixel_coords, prim_diam, self._machined_centers_trans, glass_trans)

        super().__init__(_wedge_basis, self._wedge_transmission, coefficients, as_phase, normalise)
                
    def calc_base_coeffs(self, slope_orientation, prim_diam):
        """
            Parameters:
            ----------
            slope_orientation: float
                Orientation of the wedge slope relative to the static hex pattern (radians).
            prim_diam: float
                Primary diameter of Jewel mask in meters
            Returns:
            --------
            jnp.array:
                The __base__ Tip/Tilt Zernike coefficients describing the deviation of light for this wedge.
                This will not apply the correct OPD (for the correct deviation) - this is done in apply()
                by also calculating the wavelength-dependent deviation scalar component missing here. 

                NOTE constructed this way to allow for chromatic modelling of JewelWedge. Check derivation in 
                document (ask gp).
        """
        dim_scalar_info = prim_diam/2

        # Calculate the tip/tilt coefficients that will satisfy both the slope orientation and req OPD
        # (calculated by considering OPD magnitude and direction supplied by tip and tilt polys individually,
        # treated as vector components in a circle of radius OPD)
        coeff_x = dim_scalar_info*jnp.cos(slope_orientation)/2 # Factor of 2 from zernike poly coeff for tip and tilt 
        coeff_y = dim_scalar_info*jnp.sin(slope_orientation)/2 

        coeffs = jnp.array([coeff_x, coeff_y])

        return coeffs

    def machined_centers_tfs(self, machined_centers_tf, machined_centers):
        """
            Parameters:
            ----------
            machined_centers_tf: jnp.array
                Array of transformations to apply to each hexagonal segments. Length should be equal to
                machined_centers. Size (n,3) for n machined hexagonal segments containing a [x translation,
                y translation, rotation] per segment. Units: [meters, meters, radians]
            machined_centers: jnp.array
                2D array containing the [x,y] coordinates of the centers of the machined hexagons. Units in meters.
            Returns:
            --------
            list[dl.CoordTransform]:
                List of dLux coord transformations to apply to each machined hexagonal segment to place the segment
                in the desired location on the wedge. Accounts for possible unideal translations/rotations per segment.
                Length should be equal to length of machined_centers_idxs.
        """
        if machined_centers_tf is not None:
            assert len(machined_centers_tf) == len(machined_centers), "Length of machined_centers_tf does not match length of specified machined segments"
            cen_translation = machined_centers_tf[:,0:2]
            cen_rotation = machined_centers_tf[:,2]
        else:
            cen_translation = jnp.zeros((len(machined_centers),2))
            cen_rotation = jnp.zeros(len(machined_centers))


        tf  = []
        for i, cen in enumerate(machined_centers):
                dl_tf = dl.CoordTransform(translation=cen+cen_translation[i,:], rotation = self.rot + cen_rotation[i])
                tf.append(dl_tf)

        return tf

    def machined_centers_transmission(self, rmax, machined_centers_tf, pixel_coords, prim_diam, n_pix):
        """
            Parameters:
            ----------
            rmax: float
                Max radius to vertices from center of hexagonal segment, meters.
            machined_centers_tf: jnp.array
                List of dLux coord transformations to apply to each machined hexagonal segment to place the segment
                in the desired location on the wedge. Accounts for possible unideal translations/rotations per segment.
            pixel_coords: jnp.array
                2D coordinates for each pixel center in JewelWedge (cartesian or polar)
            prim_diam: float       
                Primary diameter of Jewel mask in meters
            n_pix: int
                Number of pixels spanning vertically/horizontally over the primary mask diameter
    
            Returns:
            --------
            jnp.array:
                2D array containing the transmission values representing the machined
                hexagonal segments (1 = trasmit, 0 = block). To be used for determining
                wedge_basis.
        """
        sub_apertures = []
        for cen_tf in machined_centers_tf:
            sub_apertures.append(dl.RegPolyAperture(nsides=self.n_sides, rmax=rmax, transformation=cen_tf))

        combined_ap = dl.MultiAperture(sub_apertures) # Can apply another tf to whole pattern at this point in the future
        transmission = combined_ap.transmission(pixel_coords, prim_diam/n_pix)

        transmission = transmission.at[transmission > 1.0].set(1.0) #overlapping transmissions should cap at 1.0

        return transmission
    

    def wedge_transmission(self, pixel_coords, prim_diam, machined_centers_trans, glass_trans):
        """
            Returns:
            --------
            jnp.array:
                2D Transmission array corresponding to the entire circular wedge. Reflectance properties
                considered.
        """
        circ_wedge = dlu.circle(coords=pixel_coords, radius=0.5*prim_diam)

        if glass_trans is not None:
            circ_wedge *= glass_trans
            circ_wedge = circ_wedge.at[machined_centers_trans==1.0].set(1.0)

        return circ_wedge

    def wedge_basis(self, pixel_coords, prim_diam, machined_centers_trans):
        """
            Parameters:
            ----------
            pixel_coords: jnp.array
                2D coordinates for each pixel center in JewelWedge (cartesian or polar)
            prim_diam: float   
                Primary diameter of Jewel mask in meters
            machined_centers_trans: jnp.array
                Tranmission array corresponding to the machined centers
            
            Returns:
            --------
            jnp.array:
                The basis describing the entire circular wedge pupil with machined areas
                set to a value of 0.
        """

        # Start with basis describing entire primary diam
        circ_ap_basis = dlu.zernike_basis(js=self.zernike_idxs, coordinates=pixel_coords, diameter=prim_diam)

        # Use transmission of machined centers to set OPD/Phase to 0.0 where we want to simulate
        # light passing _thru_ the wedge
        for i, basis in enumerate(circ_ap_basis):
            ap_cut = basis.at[machined_centers_trans==1.0].set(0)
            circ_ap_basis = circ_ap_basis.at[i,:].set(ap_cut)

        return circ_ap_basis
    
    def calc_applied_deviation(self, wavelength):
        """
            Returns:
            --------
            deviation: float
                Single value representing the deviation angle (degrees) of light of a given wavelength as it passes
                through this JewelWedge
        """
        # D = A(n-1)
        deviation = self._wedge_angle * (self._material.refractive_index(wavelength=wavelength) - 1.0)
        
        return deviation
        
    
    # Override BasisOptic .apply() method to account for chromatic dispersion that JewelWedge applies
    def apply(self: dl.layers.optical_layers.OpticalLayer, wavefront: dl.wavefronts.Wavefront) -> dl.wavefronts.Wavefront:
        """
        Applies the layer (in a chromatic way) to the wavefront. 
        
        For a simple basis consisting of only tip and tilt Zernike Polynomials, where __both__ coefficients derived are 
        dependent on wedge angle formula (which is wavelength dependent), applying a wavelength-dependent deviation to 
        the wavefront is as simple as scalar multiplying the output evaluated basis. Math derviation can be found in
        documentation (ask gp).

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """

        wl = wavefront.wavelength # wavelength of wavefront (m)
        deviation = self.calc_applied_deviation(wavelength=wl) # deviation angle in degrees
        tan_dev = jnp.tan(deviation * jnp.pi/180.0) 

        wavefront *= self.transmission

        if self.as_phase:
            wavefront = wavefront.add_phase(self.eval_basis()*tan_dev)
        else:
            wavefront += (self.eval_basis()*tan_dev)

        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront


class JewelMask():
    """
        A JewelMask calculates the optical layers required for a dLux OpticalSystem.

        A JewelMask knows-a MaskPattern describing the key geometrical features of the Jewel mask, necessary 
        for the construction of JewelWedge dLux objects.

        A JewelMask also knows:
            - The number of pixels describing the diameter of the Jewel mask
            - The 2D coordinates for each pixel center in JewelMask
            - The slope orientation of each wedge in the mask (i.e. how it is clocked relative to the MaskPattern)
            - The wedge angle of each indiviudal jewel wedge
            - The Material of each wedge
            - The noll indices descibing the aberrations induced by each optical jewel wedge
            - Transmitance of the glass used in the mask: TODO if we use this functionality often, have JewelWedge calc transmittance and
                                                            add info to masterials lookup table.
            - Any machining errors in the position and rotation of the machined segments on each wedge

    """
    def __init__(self,
        mask_pattern: MaskPattern,
        n_pix: int,
        pixel_coords: jnp.array,
        wedge_angles: jnp.array,
        slope_orientations: jnp.array,
        materials: list[Material],
        zernike_noll_idxs: jnp.array = None,
        glass_trans: float = None,
        max_machining_err: jnp.array = None,
        normalise: bool=False,
        ) -> None:
        """
            Parameters:
            ----------
            MaskPattern: MaskPattern
                Object describing the general geometric features of the Jewel mask
            n_pix: int
                Number of pixels spanning vertically/horizontally over the primary diam
            pixel_coords: jnp.array
                2D coordinates for each pixel center in JewelWedge (cartesian or polar)
            wedge_angles: jnp.array
                1D array with entry i containing the angle of wedge i within the Jewel
                mask (degrees)
            slope_orientations: jnp.array
                1D array with entry i containing the rotation angle the ith wedges' slope (moves over
                a static mask pattern) in radians. Rotation starts from positive x-axis
                and moves CCW with initial slope orientation horizontal and highest point
                on the wedge furthest to the right. 
            materials: list[Material]
                List of Material objects representing the glass wedges
            zernike_noll_idxs: jnp.array = None
                1D array containing the Zernike (Noll) indices to be used to descibe JewelWedge  
                aberrations. This is seperate from the JewelWedge optical layer that encodes 
                the deflection of the wavefront. If NONE, there is no additional aberrated layer added 
                to each wedge.
            glass_trans: float
                Transmittance value for each glass wedge range [0,1]
            max_machining_err: np.array = None
                Maximum allowable positional error for each machined hexagonal segment. 
                Shape (1, 3) for n machined hexagonal segments containing a [x translation max,
                y translation max, rotation max] per segment. Units: [meters, meters, radians]

                NOTE: Errors are randomly generated but can also be set after construction of class.
            normalise : bool = False
                Whether to normalise the wavefront after passing through the JewelWedge optic.
        """
        if max_machining_err is not None:
            assert max_machining_err.shape == (1,3), """Incorrect format for machining errors. Expecting
                                                        shape of (1,3) containing 
                                                        [x translation max, y translation max, rotation max]"""
        # Priv vars
        self._MaskPattern = mask_pattern
        self._prim_diam = self._MaskPattern.primary_diam
        self._rmax = self._MaskPattern.rmax
        self._n_pix = n_pix
        self._pixel_coords = pixel_coords
        self._glass_trans = glass_trans

        assert len(wedge_angles) == self.n_wedges and len(slope_orientations) == self.n_wedges, """
            Length of wedge angles and slope orientations does not match number of wedges"""
        
        self._wedge_angles = wedge_angles
        self._slope_orientations = slope_orientations
        self._materials = materials
        self._machining_err_maxes  = max_machining_err
        self._zernike_noll_idxs = zernike_noll_idxs
        self._as_phase = False # Deviations will be applied as OPD
        self._normalise = normalise

        self._n_machined_segments = self.n_machined_segments
        self._machining_errs = self.get_rand_machining_errs
        
        
    @property
    def machining_errors(self):
        """
            Returns:
            --------
            list of np.array:
                List of arrays of shape (n,3) for n machined segments on wedge i , 
                containing the translational and rotational errors in the machining 
                of each segment of form [x_translation, y_translation, rotation].
                Translations in meters, rotations in radians.
        """
        return self._machining_errs

    @property
    def n_wedges(self):
        """
            Returns:
            --------
            int:
                Number of wedges used in the Jewel mask
        """
        return self._MaskPattern.n_wedges
    
    @property 
    def n_machined_segments(self):
        """
            Returns:    
            --------
            jnp.array:
                1D array where element i corresponds to the number of machined hexagonal segments
                on wedge i. 
        """
        machined_arr = np.array([]) 
        for i in range(self._MaskPattern.n_wedges):
            n_machined_segs = (self._MaskPattern.cart_hex_centers[self._MaskPattern.machining_sequence[i]].size)/2
            machined_arr = np.append(machined_arr, int(n_machined_segs))

        return jnp.array(machined_arr, dtype=int) 


    def set_machining_errs(self, errs_list: list[jnp.array]):
        """
            Parameters:
            ----------
            errs_list: list[jnp.array]
                List of arrays of shape (n,3) for n machined segments on wedge i , 
                containing the translational and rotational errors in the machining 
                of each segment of form [x_translation, y_translation, rotation].
                Translations in meters, rotations in radians.

            Returns:
            --------
            None
        """
        lengths = jnp.array([])
        for err_arr in errs_list:
            lengths = jnp.append(lengths, len(err_arr))  

        assert jnp.array_equal(lengths, self._n_machined_segments) == True, "Length of given machining errors does not match number of machined segments per wedge"
        
        self._machining_errs = errs_list

        return None

    @property 
    def get_rand_machining_errs(self):
        """
            Returns:    
            --------
            list of np.array:
                List of arrays of shape (n,3) for n machined segments on wedge i , 
                containing the randomly produced translation and rotational errors 
                in the machining of each segment of form [x trans, y trans, rotation].
                Translations in meters, rotiations in radians.
        """
        rtrn = None
        if self._machining_err_maxes is not None:
            x_trans_max = self._machining_err_maxes[0,0]
            y_trans_max = self._machining_err_maxes[0,1]
            rot_max = self._machining_err_maxes[0,2]

            machining_errs = [] 
            for i in range(self._MaskPattern.n_wedges):
                # Randomly generate positional errors for each machined hexagonal segment
                rand_n = np.random.rand(self._n_machined_segments[i],3) # Rand (0,1)
                rand_n[:,0] = -x_trans_max + rand_n[:,0]*(x_trans_max - -x_trans_max) # re-map to [-max, max]
                rand_n[:,1] = -y_trans_max + rand_n[:,1]*(y_trans_max - -y_trans_max)
                rand_n[:,2] = -rot_max + rand_n[:,2]*(rot_max - -rot_max)
                rand_n = jnp.asarray(rand_n) # use Jax (no random.rand in Jax numpy)

                machining_errs.append(rand_n)

            rtrn = machining_errs

        return rtrn 

    @property 
    def jewel_layers(self):
        """
            Returns:
            --------
            list of tuples:
                List of tuples of form (name: str, optic: JewelWedge).
                Definition compatible with dLux layers format.
        """
        # Set rotation of segments according to MaskPattern
        # (default is pi/6 in JewelWedge (class variable) unless changed)
        JewelWedge.rot = self._MaskPattern.seg_rotation

        # Begin with shim layer. This is just a simple dLux Transmissive layer
        # NOTE this layer will always be assumed to be machined perfectly (i.e. no tf's applied)
        # We can use a dummy JewelWedge (with all hexagonal segments machined) to retrieve this info
        layers = []
        shim = JewelWedge(wedge_angle= self._wedge_angles[0],
                        slope_orientation= self._slope_orientations[0],
                        material= self._materials[0],
                        prim_diam = self._prim_diam, 
                        rmax = self._rmax, 
                        n_pix = self._n_pix, 
                        pixel_coords=self._pixel_coords, 
                        hex_centers = self._MaskPattern.cart_hex_centers, 
                        machined_centers_idxs=np.arange(self._MaskPattern.n_tilings))
        shim = shim._machined_centers_trans
        
        layers.append(("Shim", dl.layers.TransmissiveLayer(shim)))

        # Add JewelWedge layers (along with a corresponding Jewel Wedge aberration layer if specified)
        if self._machining_errs is not None:
            machining_errors = self._machining_errs
        else:
            machining_errors = [None] *  self.n_wedges

        for i in range(self._MaskPattern.n_wedges):
                optic = JewelWedge(wedge_angle= self._wedge_angles[i],
                                slope_orientation= self._slope_orientations[i],
                                material= self._materials[i],
                                prim_diam = self._prim_diam, 
                                rmax = self._rmax, 
                                n_pix = self._n_pix, 
                                pixel_coords=self._pixel_coords, 
                                hex_centers = self._MaskPattern.cart_hex_centers, 
                                machined_centers_idxs=self._MaskPattern.machining_sequence[i], 
                                glass_trans= self._glass_trans,
                                machined_centers_tf=machining_errors[i], 
                                as_phase=self._as_phase, 
                                normalise=self._normalise)
                strname = "Wedge" + str(i)
                layers.append((strname, optic))

                # Separate layer to describe aberrations on this Jewel wedge
                if self._zernike_noll_idxs is not None:
                    coeffs = np.zeros(self._zernike_noll_idxs.shape) # Initalise to 0. I.e. no aberrations
                    zernike_bases = dlu.zernike_basis(js = self._zernike_noll_idxs, coordinates=self._pixel_coords, diameter=self._prim_diam)
                    
                    # Force 0 OPD for machined areas
                    for j, basis in enumerate(zernike_bases):
                        ap_cut = basis.at[optic._machined_centers_trans==1.0].set(0.0)
                        zernike_bases = zernike_bases.at[j,:].set(ap_cut)

                    aberrated_optic = dl.layers.BasisLayer(basis=zernike_bases, coefficients=coeffs)
    
                    strname = "WedgeAberration" + str(i)
                    layers.append((strname, aberrated_optic))

     
        return layers

class JewelAngularOpticalSystem(dl.optical_systems.ParametricOpticalSystem, dl.optical_systems.LayeredOpticalSystem):
    """
        JewelAngularOpticalSystem is-a dLux LayeredOpticalSystem which can model
        the propagation of a wavefront to an image plane with `psf_pixel_scale` 
        in units of arcseconds. 
        
        JewelAngularOpticalSystem achieves almost the same functionality as dl.AngularOpticalSystem
        except that it can also return the result of the propgation from various rectangular areas
        on the detector, defined by pixel bounds (useful for optimsation on individual inteferograms).
        TODO for model method (not just propagate_mono)
    """

    source: dl.sources.Source

    _psf_shifts: np.array = eqx.field(static=True) # NOTE classes inherited are frozen, so this class is too. 
                                                    # As such, _psf_bounds can only be set once in class constructor
                                                    #field not to interact with any jax tf
    _sub_psf_npixels: int = eqx.field(static=True)

    def __init__(
        self: dl.optical_systems.OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list[dl.layers.optical_layers.OpticalLayer, tuple],
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
        psf_shifts: np.array = None,
        sub_psf_npixels: int = None,
        source: dl.sources.Source = None,
        ):
        """
        Parameters
        ----------
        wf_npixels : int
            The number of pixels representing the wavefront.
        diameter : Array, metres
            The diameter of the initial wavefront to propagate.
        layers : list[OpticalLayer, tuple]
            A list of `OpticalLayer` transformations to apply to wavefronts. The list
            entries can be either `OpticalLayer` objects or tuples of (key, layer) to
            specify a key for the layer in the layers dictionary.
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float, arcseconds
            The pixel scale of the final PSF in units of arcseconds.
        oversample : int
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        psf_shifts : jnp.array = None
            (n,2) Array consisting of the pixel [x,y] shifts to the psf to position the n
            sub-psf arrays. 
            NOTE: compatible with Wavefront propagate method definition of shift:
                     +ve pixel shift in x direction will shift center to left
                     +ve pixel shift in y direction will shift center down
        sub_psf_npixels: int = None
            Number of pixels in the sub-psf arrays. Assuming square sub-psf arrays for now
        source : Source = None
            The source to .model() the system with.

        """
        if psf_shifts is not None:
            assert psf_shifts.shape[-1] == 2, "Incorrect format for psf shifts. Expecting (n,2) where n is the number of sub-psf arrays."
            assert sub_psf_npixels is not None, "Sub-pixel dimensions not defined."
        self._psf_shifts = psf_shifts
        self._sub_psf_npixels = sub_psf_npixels

        self.source = source

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )
            
    def calc_sub_psf(
        self,
        wavelength: Array,
        offset: Array = jnp.zeros(2),
        shift: Array = None,
        ):

        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        pixel_scale = dlu.arcsec2rad(self.psf_pixel_scale / self.oversample)
        wf = wf.propagate(npixels = self._sub_psf_npixels, 
                    pixel_scale=pixel_scale,
                    shift = shift,
                    pixel = True,
                    )
        
        return wf.psf

    def propagate_mono(
        self: dl.optical_systems.OpticalSystem,
        wavelength: Array,
        offset: Array = jnp.zeros(2),
        return_wf: bool = False,
        return_sub_psf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_sub_psf: bool = False
            If return_wf is False, setting return_sub_psf to True will return list of n PSF
            arrays corresponding to the n sub-regions defined by psf_bounds.
        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False AND 'return_sub_psf' is False, returns the psf Array.
            if `return_wf` is False AND 'return_sub_psf' is True, returns the psf Array
            per sub-region on the detector.
            if `return_wf` is True, returns the Wavefront object (regardless of 'return_sub_psf' setting).
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True) # Calling LayeredOpticalSystem propagate_mono

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        to_return = 0 # Avoid multiple returns
        if return_wf:
            to_return = wf
        elif return_sub_psf:
            batched_get_sub_psfs = vmap(self.calc_sub_psf, in_axes=(None,None,0))
            to_return = batched_get_sub_psfs(wavelength, offset, self._psf_shifts)
        else:
            to_return = wf.psf

        return to_return

    def model(self):
        """
            Returns:
            --------
            psf: np.array
                Output PSF array corresponding to optical response to a dLux defined source object
                
        """
        psf = super().model(source=self.source, return_wf=False, return_psf=False)

        return psf

def rotation_matrix(theta: float):
    """
        Parameters:
        ----------
        theta: float
            Angle of rotation in radians
        Returns:
        --------
        jnp.array:
            2D rotation matrix (CCW) for given angle
    """
    return jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])