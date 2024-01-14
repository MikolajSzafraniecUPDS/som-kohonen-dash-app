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
        self.RGB_vals = rgb_vals

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


class SelfOrganizingMap:

    def __init__(
            self,
            size: int = 250,
            include_alpha_channel: bool = True,
            initial_neighbourhood_radius: float = 50.0,
            initial_learning_rate: float = 1.0,
            decay_lambda: float = 20
    ):
        """
        Create an instance of self organizing map (Kohonen network).

        :param size: number of neurons per row/column (our network is a square,
            so the actual size will be size x size
        :param include_alpha_channel: should alpha channel be included
        :param initial_neighbourhood_radius: initial neighbourhood radius - it
            decreases after each iteration of learning
        :param initial_learning_rate: initial value of learning rate
        :param decay_lambda: parameter used in the formula decreasing neighbourhood
            radius and learning rate over time. The higher, the slower the
            neighbourhood radius and learning rate decrease
        """
        self.neuron_list = None
        self.size = size
        self.initial_neighbourhood_radius = initial_neighbourhood_radius
        self.initial_learning_rate = initial_learning_rate
        self.include_alpha_channel = include_alpha_channel
        self.decay_lambda = decay_lambda
        self.current_iteration = 0
        self._init_neurons()

    def _init_neurons(self) -> None:
        """
        Create a list of neurons. They will be initialized with a random
        RGB(A) values
        """
        neuron_list = [
            Neuron(i, j, self.include_alpha_channel)
            for i in range(self.size) for j in range(self.size)
        ]
        self.neuron_list = neuron_list

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
        if (self.neuron_list is not None) and (current_val != value):
            self._init_neurons()

    include_alpha_channel = property(
        _get_alpha_channel_indicator,
        _set_alpha_channel_indicator
    )

    def _get_current_neighbourhood_radius(self) -> float:
        """
        Calculate current value of neighbourhood radius, based on
        initial value, number of iteration and decay lambda. Formula
        described in the article: https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da

        :return: current value of neighbourhood radius
        """
        res = self.initial_neighbourhood_radius*np.exp(
            -self.current_iteration / self.decay_lambda
        )
        return res

    def _get_current_learning_rate(self) -> float:
        """
        Calculate current value of learning rate, based on
        initial value, number of iteration and decay lambda.
        Formula described in the article: https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da

        :return: current value of learning rate
        """
        res = self.initial_learning_rate*np.exp(
            -self.current_iteration / self.decay_lambda
        )
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
