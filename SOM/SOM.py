import numpy as np


class Neuron:

    """
    Class representing single neuron of Kohonen network (Self-organising map).
    It stores x-axis and y-axis indices of neuron and RGB/RGBA values assigned to it
    (as numpy array)
    """

    def __init__(self, idx_x: int, idx_y: int, include_alpha_channel: bool = False):
        """
        Create an instance of the Neuron class

        :param idx_x: index on network's x axis
        :param idx_y: index on network's y axis
        :param include_alpha_channel: bool - is network of RGBA type. If
            False only 3 values (RGB) will be assigned
        """
        self.idx_x = idx_x
        self.idx_y = idx_y
        self.include_alpha_channel = include_alpha_channel
        self.RGB_vals = self._initialize_rgb_values()

    def _initialize_rgb_values(self) -> np.ndarray:
        """
        Assign initial RGB(A) values, picked randomly
        from range (0, 255)

        :return: numpy array containing 3 (RGB) or 4 (RGBA)
            integer values
        """
        vals_num: int = 4 if self.include_alpha_channel else 3
        rgb_vals = np.random.randint(
            low=0, high=255, size=vals_num
        )
        return rgb_vals
