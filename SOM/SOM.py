import os
import numpy as np
import logging.config
from PIL import Image

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
        current_weights_input_vec_diff = self.RGB_vals - input_vector
        new_weights = self.RGB_vals + learning_rate*neighbourhood_value*current_weights_input_vec_diff
        new_weights = new_weights.astype(np.uint8)
        self.RGB_vals = new_weights


class SelfOrganizingMap:

    def __init__(
            self,
            size: int = 250,
            include_alpha_channel: bool = True,
            initial_neighbourhood_radius: float = 50,
            initial_learning_rate: float = 100,
            decay_lambda: float = 0.1,
            neighbourhood_type: str = "gaussian"
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
        :param neighbourhood_type: type of neighbourhood function to use - one
            of 'gaussian' or 'square'
        """
        self.neurons = None
        self.size = size
        self.initial_neighbourhood_radius = initial_neighbourhood_radius
        self.initial_learning_rate = initial_learning_rate
        self.include_alpha_channel = include_alpha_channel
        self.decay_lambda = decay_lambda
        self.current_iteration = 0
        self.neighbourhood_type = neighbourhood_type
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

    def _get_neighbourhood_type(self) -> str:
        """
        Getter for 'neighbourhood_type' property

        :return: neighbourhood_type value
        """
        return self._neighbourhood_type

    def _set_neighbourhood_type(self, value: str) -> None:
        """
        Setter for 'neighbourhood_type' property. It verifies whether
        proper value is trying to be set.

        :param value: neighbourhood type to set
        """
        if value not in ["gaussian", "square"]:
            raise ValueError("Neighbourhood type must be one of ['gaussian', 'square']")
        self._neighbourhood_type = value

    neighbourhood_type = property(
        _get_neighbourhood_type,
        _set_neighbourhood_type
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
        # Metoda z R-a:
        #res = self.initial_neighbourhood_radius/(np.exp(self.decay_lambda*self.current_iteration))
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
        # Metoda z R-a:
        #res = self.initial_learning_rate / (np.exp(self.decay_lambda * self.current_iteration))
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
            low=0, high=255, size=vals_num
        )
        bmu = self._get_bmu(input_vector)
        current_learning_rate = self._get_current_learning_rate()
        neighbourhood_func = self._gaussian_neighbourhood if self.neighbourhood_type == "gaussian" \
            else self._square_neighbourhood
        for neuron in self.neurons.values():
            neighbourhood_value = neighbourhood_func(bmu, neuron)
            neuron.update_weights(
                input_vector=input_vector,
                learning_rate=current_learning_rate,
                neighbourhood_value=neighbourhood_value
            )
        self.current_iteration += 1

    def train_network(self, number_of_iterations: int) -> None:
        for i in range(number_of_iterations):
            self.train_network_single_iteration()
            logger.info(
                "Training iteration number {0} passed successfully".format(i+1)
            )

    def reset_network(self) -> None:
        """
        Reset network (reinitialize neurons, set number of iteration to zero).
        """
        self.current_iteration = 0
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
