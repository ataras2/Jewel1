"""
    Class to read IDL .dat file with information regarding hexagonal
    tiling patterns in Jewel mask
"""
import numpy as np
import pandas as pd
from scanf import scanf

class IDLFileReader:
    """
        This class knows the .dat file produced by an IDL program responsible for Jewel mask pattern 
        creations.

        IDLFileReader extracts the relevant info from the .dat file to be used for later description
        of a Jewel mask optical system, defined by _read_header() and _read_solutions() using
        agreed upon conventions. Can change these functions if file format of .dat 
        file changes.
    """
    def __init__(self, fname: str) -> None:
        """
            Read Jewel pattern information produced by previous IDL prog

            Parameters:
            -----------
            fname: str
                String to filepath of IDL file
        """
        # Private vars
        self._n_tilings, self._n_seg, self._is_centered, self._idim, self._jdim, self._used_segs = self._read_header(fname)
        
        self._solution_array, self._merit = self._read_solutions(fname)

        ## can add more info just basics for now

    def _read_header(self, fname):
        with open(fname, "r", encoding="utf-8") as f:
            header1 = f.readline()
            header2 = f.readline()
            header3 = f.readline()
            header4 = f.readline()

            # use scanf
            h1_res = scanf("%dx%d", header1)
            h2_res = scanf("Centering: %s", header2)
            h3_res = scanf("idim: %d", header3)
            h3_res2 = scanf("jdim: %d", header3)
            h4_res = scanf("used segs: %s", header4)

            for val in h1_res:
                if val < 0:
                    raise ValueError(
                        f"Number of tilings and segments must be positive, not {val}"
                    )
            if h2_res[0] not in ["True", "False"]:
                raise ValueError(
                    f"Centering must be True or False, not {h2_res[0]}"
                )
            if h3_res[0] < 0:
                raise ValueError(
                    f"Number of segments must be positive, not {h3_res[0]}"
                )
            
            if h3_res2 is not None:
                if h3_res2[0] < 0:
                    raise ValueError(
                        f"Number of segments must be positive, not {h3_res2[0]}"
                    )
        

            n_tilings = int(h1_res[0])
            n_seg = int(h1_res[1])

            is_centered = h2_res[0] == "True"

            used_segs = [int(x) for x in h4_res[0].split(",")]

            idim = int(h3_res[0])

            if h3_res2 is not None:
                jdim = int(h3_res2[0])
            else:
                jdim = None

        return n_tilings, n_seg, is_centered, idim, jdim, used_segs

    def _read_solutions(self, fname):
        """Reads in the .dat file and converts it to a numpy array of intergers

        Args:
            file_path (str): path to the file

        Returns:
            (np.array): A numpy array of integers
        """
        file_data = pd.read_csv(
            fname,
            header=None,  # No header on dataframe
            skip_blank_lines=True,  # Skip blank lines
            skiprows=4,  # Skips first row of file (size of the header)
        )
        file_array = file_data[0].str.split(
            expand=True
        )  # Splits all the data into columns
        main_array = np.array(
            file_array, dtype=float
        )  # Converts to Numpy array

        n_solutions = int(main_array.shape[0] / (self._n_seg + 1))

        s_count = np.zeros(n_solutions, dtype=np.uint32)
        s_merit = np.zeros(n_solutions, dtype=float)
        s_index = np.zeros(
            (n_solutions, self._n_seg, main_array.shape[1]), dtype=np.uint32
        )

        # parse the data into the counter (int), the final merit (float), and the index of pattern segments
        for i in range(n_solutions):
            s_count[i] = main_array[i * (self._n_tilings + 1), 0]
            s_merit[i] = main_array[i * (self._n_tilings + 1), 1]
            s_index[i, :, :] = main_array[
                i * (self._n_seg + 1) + 1 : i * (self._n_seg + 1) + 1 + self._n_seg, :
            ]

        return s_index, s_merit

    def get_hex_centers_idxs(self, sol_idx: int):
        """
            Return 1D array of solutions for hex centers idx's on a triangular grid
            (from bottom to top, left to right) according to the sol_idx solution

            Parameters:
            -----------
            sol_idx: int
                Index of the solution array

            Returns:
            ----------
            sol: 2D np.array
                Center idx's for hexagon on triganular grid. Shape (n_tilings, n_seg), where
                row i is the center idx's for the ith tiling pattern
        """
        assert sol_idx < len(self._merit), "Index outside of available solutions {}".format(len(self._merit))

        sol = self._solution_array[sol_idx,:]

        return np.transpose(sol)
    
    @property
    def pattern_info(self):
        """
            Returns:
            --------
            int: n_tilings
                Number of unique tilings/patterns listed in IDL header
            int: n_seg
                Number of hexagonal segments in each unique tiling pattern as given
                in IDL header
            int: idim
                Underlying dimensional grid (unitless) as given by IDL header
        """
        return self._n_tilings, self._n_seg, self._idim, self._jdim
    
