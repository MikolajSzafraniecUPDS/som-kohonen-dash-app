"""
Definitions of neuron and self-organising map. Details about learning parameters
(learning rate, neighbourhood functions, etc.) can be found in the following
article: https://ijmo.org/vol6/504-M08.pdf
"""


import os
import numpy as np
import logging.config
from PIL import Image
from typing import Tuple
from strenum import StrEnum

logging.config.fileConfig(os.path.join("config", "logging.conf"))
logger = logging.getLogger("consoleLogger")


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
        self.RGB_vals = None
        self.idx_x = idx_x
        self.idx_y = idx_y
        self.include_alpha_channel = include_alpha_channel
        self.initialize_rgb_values()

    def initialize_rgb_values(self) -> None:
        """
        Assign RGB(A) values, picked randomly
        from range (0, 255)
        """
        vals_num: int = 4 if self.include_alpha_channel else 3
        rgb_vals = np.random.randint(
            low=0, high=255, size=vals_num
        )
        self.RGB_vals = rgb_vals.astype(np.uint8)

    def calc_distance(self, vec: np.ndarray) -> float:
        """
        Calc Euclidean distance between assigned RGB(A) values
        and given vector.

        :param vec: the vector for which the distance is to be calculated
        :return: results as float number
        """
        res = np.linalg.norm(
            self.RGB_vals - vec
        )
        
        return res

    def update_weights(
            self,
            input_vector: np.ndarray,
            learning_rate: float,
            neighbourhood_value: float
    ) -> None:
        """
        Update neuron's weights (RGB[A]) values) based on learning input
        vector, current learning rate and neighbourhood value to the BMU.

        :param input_vector: input learning vector
        :param learning_rate: current learning rate
        :param neighbourhood_value: neighbourhood value calculated for given neuron
            and BMU indices
        """
        current_weights_input_vec_diff = input_vector-self.RGB_vals
        new_weights = self.RGB_vals + (learning_rate*neighbourhood_value*current_weights_input_vec_diff)
        new_weights = new_weights.astype(np.uint8)
        self.RGB_vals = new_weights


class NeighbourhoodType(StrEnum):
    """
    Available neighbourhood functions
    """
    GAUSSIAN = "gaussian"
    BUBBLE = "bubble"


# Available types of learning rate decay function
class LearningRateDecay(StrEnum):
    """
    Available learning rate decay functions
    """
    LINEAR = "linear"
    INVERSE_OF_TIME = "inverse_of_time"
    POWER_SERIES = "power_series"


class SelfOrganizingMap:

    def __init__(
            self,
            size: int = 100,
            include_alpha_channel: bool = True,
            initial_neighbourhood_radius: float = 0.1,
            initial_learning_rate: float = 0.5,
            neighbourhood_type: NeighbourhoodType = NeighbourhoodType.GAUSSIAN,
            learning_rate_decay_func: LearningRateDecay = LearningRateDecay.INVERSE_OF_TIME,
            rgba_low: Tuple[int] = (0, 0, 0, 0),
            rgba_high: Tuple[int] = (256, 256, 256, 256)
    ):
        """
        Create an instance of self organizing map (Kohonen network).

        :param size: number of neurons per row/column (our network is a square,
            so the actual size will be size x size
        :param include_alpha_channel: should alpha channel be included
        :param initial_neighbourhood_radius: initial value of neighbourhood parameter
            as a percentage of network radius - must be in range 0 < inr <= 1
        :param initial_learning_rate: initial value of learning rate - must be a float
            in range 0 < lr <= 1
        :param neighbourhood_type: type of neighbourhood function to use - one
            of 'gaussian' or 'bubble'
        :param learning_rate_decay_func: type of decay function for learning rate. Possible
            choices are 'linear', 'inverse_of_time' and 'power_series'. Details in paper
            https://ijmo.org/vol6/504-M08.pdf
        :param rgba_low: lower limit of the RGBA values to drawn for each learning iteration.
            Must be a tuple of length 4 with values in range [0, 255]
        param: rgba_high: upper limit of the RGBA values to drawn for each learning iteration.
            Must be a tuple of length 4 with values in range [1, 256] (np.random.randint function
            uses 'half-open' interval in form [low, high). Details in documentation:
            https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
        """
        self._LEARNING_RATE_DECAY_FUNCTIONS = {
            "linear": self._get_learning_rate_linear,
            "inverse_of_time": self._get_learning_rate_inverse_of_time,
            "power_series": self._get_learning_rate_power_series
        }
        self._NEIGHBOURHOOD_FUNCTIONS = {
            "gaussian": self._gaussian_neighbourhood,
            "bubble": self._square_neighbourhood
        }
        self.neurons = None
        self.size = size
        self.initial_neighbourhood_radius = initial_neighbourhood_radius
        self.initial_learning_rate = initial_learning_rate
        self.include_alpha_channel = include_alpha_channel
        self.current_iteration = 1
        self.number_of_iterations = 200
        self.neighbourhood_type = neighbourhood_type
        self.learning_rate_decay_func = learning_rate_decay_func
        self.rgba_low = rgba_low
        self.rgba_high = rgba_high
        self._init_neurons()

    def _init_neurons(self) -> None:
        """
        Create a dict containing network's neurons. They will be
        initialized with a random RGB(A) values
        """
        neurons = {
            (i, j): Neuron(i, j, self.include_alpha_channel)
            for i in range(self.size) for j in range(self.size)
        }
        self.neurons = neurons

    def _get_alpha_channel_indicator(self) -> bool:
        """
        Getter for 'include_alpha_channel' property

        :return: include_alpha_channel value
        """
        res = self._include_alpha_channel if hasattr(self, '_include_alpha_channel') else None
        return res

    def _set_alpha_channel_indicator(self, value: bool) -> None:
        """
        Setter for 'include_alpha_channel' property. In case
        when the field is updated with different value than the
        current one we need to reinstantiate all the neurons

        :param value: value to set as 'include_alpha_channel' property
        """
        current_val = self.include_alpha_channel
        self._include_alpha_channel = value
        # Reinstantiate neurons in case when they were already created
        # and new value of 'include_alpha_channel' is different from
        # the current one
        if (self.neurons is not None) and (current_val != value):
            self.reset_network()

    include_alpha_channel = property(
        _get_alpha_channel_indicator,
        _set_alpha_channel_indicator
    )

    def _get_size(self) -> int:
        """
        Size property getter

        :return: size of the network
        """
        res = self._size if hasattr(self, '_size') else None
        return res

    def _set_size(self, value: int) -> None:
        """
        Size property setter. If the value is different from the current one
        network is reset

        :param value: size of network
        """
        current_size = self.size
        self._size = value
        if (self.neurons is not None) and (current_size != value):
            self.reset_network()

    size = property(
        _get_size,
        _set_size
    )

    def resize_and_update_alpha_channel_indicator(
            self, new_size: int, alpha_channel_ind: bool
    ) -> None:
        """
        In some cases we want to update both size and alpha channel
        indicator at the same time. To not reset network twice (it
        would be done if we would assign new size as a first step
        and new alpha channel indicator as a second step) we can
        use this method.

        :param new_size: new size of network
        :param alpha_channel_ind: new alpha channel indicator
        """
        self._size = new_size
        self._include_alpha_channel = alpha_channel_ind
        self.reset_network()

    def _get_initial_neighbourhood_radius(self) -> float:
        """
        Get initial neighbourhood factor as a ratio of network
        radius

        :return: initial neighbourhood radius
        """
        return self._initial_neighbourhood_radius_ratio

    def _set_initial_neighbourhood_radius(self, value: float) -> None:
        """
        Set initial neighbourhood radius ratio and convert it to
        actual radius according to the network size. Value must be in
        range 0 < value <= 1

        :param value: initial neighbourhood radius ratio
        """
        if (value <= 0) or (value > 1):
            raise ValueError(
                "Initial neighbourhood radius must be in range 0 < value <= 1"
            )
        self._initial_neighbourhood_radius_ratio = value
        self._initial_neighbourhood_radius = int(self.size*value)

    initial_neighbourhood_radius = property(
        _get_initial_neighbourhood_radius,
        _set_initial_neighbourhood_radius
    )

    def _get_initial_learning_rate(self) -> float:
        """
        Get initial learning rate

        :return:
        """
        return self._initial_learning_rate

    def _set_initial_learning_rate(self, value: float) -> None:
        """
        Set initial learning rate. Value must be in range
        0 < value <= 1

        :param value: initial learning rate to assign
        """
        if (value <= 0) or (value > 1):
            raise ValueError(
                "Initial learning rate must be in range 0 < value <= 1."
            )
        self._initial_learning_rate = value

    initial_learning_rate = property(
        _get_initial_learning_rate,
        _set_initial_learning_rate
    )

    def _get_rgba_low(self) -> Tuple[int]:
        """
        Getter for 'rgba_low' property

        :return: rgba_low value
        """
        return self._rgba_low

    def _set_rgba_low(self, value: Tuple[int]) -> None:
        """
        Setter for 'rgba_low' property, validating its length
        and range of values.

        :param value: value of rgba_low property to set
        """
        if len(value) != 4:
            raise ValueError(
                "Length of rgba_low tuple must be 4."
            )

        if not all([lb >= 0 for lb in value]):
            raise ValueError(
                "Lower limit of RGBA values must be equal to or higher than 0."
            )

        if not all([lb < 256 for lb in value]):
            raise ValueError(
                "Lower limit of RGBA values must lower than 256."
            )

        self._rgba_low = value

    rgba_low = property(
        _get_rgba_low,
        _set_rgba_low
    )

    def _get_rgba_high(self) -> Tuple[int]:
        """
        Getter for 'rgba_high' property

        :return: rgba_high value
        """
        return self._rgba_high

    def _set_rgba_high(self, value: Tuple[int]) -> None:
        """
        Setter for 'rgba_high' property, validating its length
        and range of values.

        :param value: value of rgba_high property to set
        """
        if len(value) != 4:
            raise ValueError(
                "Length of rgba_high tuple must be 4."
            )

        if not all([lb > 0 for lb in value]):
            raise ValueError(
                "Upper limit of RGBA values must be higher than 0."
            )

        if not all([lb < 257 for lb in value]):
            raise ValueError(
                "Upper limit of RGBA values must lower than 257."
            )

        self._rgba_high = value

    rgba_high = property(
        _get_rgba_high,
        _set_rgba_high
    )


    def _get_current_neighbourhood_radius(self) -> float:
        """
        Calculate current value of neighbourhood radius, based on
        initial value, number of current iteration and total number of
        iterations. Formula described in the paper: https://ijmo.org/vol6/504-M08.pdf

        :return: current value of neighbourhood radius
        """
        res = self._initial_neighbourhood_radius*np.exp(
            -self.current_iteration / self.number_of_iterations
        )
        return res

    def _get_learning_rate_linear(self) -> float:
        """
        Calculate current value of learning rate using linear
        decay function. Details in the paper: https://ijmo.org/vol6/504-M08.pdf

        :return: current value of learning rate
        """
        res = self.initial_learning_rate * (1/self.current_iteration)
        return res

    def _get_learning_rate_inverse_of_time(self) -> float:
        """
        Calculate current value of learning rate using inverse of
        time function. Details in the paper: https://ijmo.org/vol6/504-M08.pdf

        :return: current value of learning rate
        """
        res = self.initial_learning_rate*(1-(self.current_iteration/self.number_of_iterations))
        return res

    def _get_learning_rate_power_series(self) -> float:
        """
        Calculate current value of learning rate using power series
        function. Details in the paper: https://ijmo.org/vol6/504-M08.pdf

        :return: current value of learning rate
        """
        res = self.initial_learning_rate*np.exp(self.current_iteration/self.number_of_iterations)
        return res

    def _get_current_learning_rate(self) -> float:
        """
        Calculate current value of learning rate

        :return: current value of learning rate
        """
        learning_rate_func = self._LEARNING_RATE_DECAY_FUNCTIONS.get(
            self.learning_rate_decay_func
        )
        res = learning_rate_func()
        return res

    @staticmethod
    def _get_neuron_distance(neuron_1: Neuron, neuron_2: Neuron) -> float:
        """
        Calculate distance between neurons based on their indices

        :param neuron_1: neuron to calculate distance for
        :param neuron_2: neuron to calculate distance for
        :return: Euclidean distance between neurons
        """
        ind_1_array = np.array([
            neuron_1.idx_y,
            neuron_1.idx_x
        ])
        ind_2_array = np.array([
            neuron_2.idx_y,
            neuron_2.idx_x
        ])
        res = np.linalg.norm(ind_1_array-ind_2_array)
        return res

    def _gaussian_neighbourhood(self, neuron_1: Neuron, neuron_2: Neuron) -> float:
        """
        Gaussian neighbourhood function. It allows to calculate the value of
        neuron's neighbourhood for given iteration of learning process.
        Formula described in the article: https://home.agh.edu.pl/~vlsi/AI/koho_t/

        :param neuron_1: neuron to calculate neighbourhood value for
        :param neuron_2: neuron to calculate neighbourhood value for

        :return: value of neighbourhood according to the gaussian formula
        """
        neurons_distance = self._get_neuron_distance(neuron_1, neuron_2)
        neighbourhood_radius = self._get_current_neighbourhood_radius()
        res = np.exp(
            -neurons_distance / (2*(neighbourhood_radius**2))
        )

        return res

    def _square_neighbourhood(self, neuron_1: Neuron, neuron_2: Neuron) -> float:
        """
        Square neighbourhood function. It allows to calculate the value of
        neuron's neighbourhood for given iteration of learning process.
        Formula described in the article: https://home.agh.edu.pl/~vlsi/AI/koho_t/

        :param neuron_1: neuron to calculate neighbourhood value for
        :param neuron_2: neuron to calculate neighbourhood value for

        :return: value of neighbourhood according to the square formula
        """
        neurons_distance = self._get_neuron_distance(neuron_1, neuron_2)
        neighbourhood_radius = self._get_current_neighbourhood_radius()
        res = 1.0 if neurons_distance <= neighbourhood_radius else 0.0
        return res

    def _get_bmu(self, input_vector: np.ndarray) -> Neuron:
        """
        Get BMU (best matching unit) for given input vector.

        :param input_vector: input learning vector for given iteration
        :return: BMU neuron
        """
        bmu = None
        current_min = np.Inf
        for neuron in self.neurons.values():
            neuron_dist = neuron.calc_distance(input_vector)
            if neuron_dist < current_min:
                current_min = neuron_dist
                bmu = neuron

        return bmu

    def train_network_single_iteration(self) -> None:
        """
        Single iteration of network's training.
        """
        vals_num = 4 if self.include_alpha_channel else 3
        input_vector = np.random.randint(
            low=self.rgba_low[:vals_num],
            high=self.rgba_high[:vals_num],
            size=vals_num
        )
        bmu = self._get_bmu(input_vector)
        current_learning_rate = self._get_current_learning_rate()
        neighbourhood_func = self._NEIGHBOURHOOD_FUNCTIONS.get(
            self.neighbourhood_type
        )
        for neuron in self.neurons.values():
            neighbourhood_value = neighbourhood_func(bmu, neuron)
            neuron.update_weights(
                input_vector=input_vector,
                learning_rate=current_learning_rate,
                neighbourhood_value=neighbourhood_value
            )
        self.current_iteration += 1

    def train_network(self, number_of_iterations: int) -> None:
        self.number_of_iterations = number_of_iterations
        for i in range(number_of_iterations):
            self.train_network_single_iteration()
            logger.info(
                "Training iteration number {0} passed successfully".format(i+1)
            )

    def reset_network(self) -> None:
        """
        Reset network (reinitialize neurons, set number of iteration to zero).
        """
        self.current_iteration = 1
        self._init_neurons()

    def _neuron_weights_to_array(self) -> np.ndarray:
        """
        Create 3D numpy array and insert neuron weights to it. Array can be
        transformed rendered as RGB(A) image.

        :return: numpy array containing neurons weights
        """
        rgb_num_vals = 4 if self.include_alpha_channel else 3
        output_array = np.zeros(
            [self.size, self.size, rgb_num_vals],
            dtype=np.uint8
        )
        for neuron in self.neurons.values():
            output_array[neuron.idx_y, neuron.idx_x, :] = neuron.RGB_vals

        return output_array

    def to_image(self) -> Image.Image:
        """
        Get Image representing current state of network

        :return: Image object representing neurons RGB(A) values
        """
        network_array = self._neuron_weights_to_array()
        res = Image.fromarray(network_array)
        return res
