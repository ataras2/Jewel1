"""Utils for hexagon related calculations"""

import numpy as np


class HexMath:
    """Collection of methods for working with hexagons

    unit invariant
    """

    def __init__(self, **kwargs) -> None:
        """Given one parameter from
            side_length,
            incircle_r,
            circum_radius,
            long_diagonal,
            short_diagonal,
        store a representation of a hexagon

        Raises:
            ValueError: if 0 or more than one parameters are given
        """
        if len(kwargs) != 1:
            raise ValueError("Must supply exactly one parameter")

        allowed_values = [
            "side_length",
            "incircle_r",
            "circum_radius",
            "long_diagonal",
            "short_diagonal",
        ]

        given_param = list(kwargs.keys())[0]
        assert given_param in allowed_values

        if given_param == "side_length":
            self.side_length = kwargs["side_length"]
        elif given_param == "incircle_r":
            self.side_length = kwargs["incircle_r"] / (np.sqrt(3) / 2)
        elif given_param == "circum_radius":
            self.side_length = kwargs["circum_radius"]
        elif given_param == "long_diagonal":
            self.side_length = kwargs["long_diagonal"] / 2
        elif given_param == "short_diagonal":
            self.side_length = kwargs["short_diagonal"] / np.sqrt(3)

    @property
    def area(self):
        """
        Returns:
            float: Area of the hexagon
        """
        return 3 * np.sqrt(3) / 2 * self.side_length**2

    @property
    def perimeter(self):
        """
        Returns:
            float: Perimeter of the hexagon
        """
        return 6 * self.side_length

    @property
    def incircle_r(self):
        """
        Returns:
            float: Radius of the largest circle that
                is completly within the hexagon
        """
        return self.side_length * np.sqrt(3) / 2

    @property
    def circum_radius(self):
        """
        Returns:
            float: Radius of the smallest circle that
                completly encloses the hexagon
        """
        return self.side_length

    @property
    def long_diagonal(self):
        """
        Returns:
            float: Diagonal from a vertex to its opposite
        """
        return 2 * self.side_length

    @property
    def short_diagonal(self):
        """
        Returns:
            float: Diagonal from a vertex to a
                vertex other than the one opposite
        """
        return self.side_length * np.sqrt(3)
