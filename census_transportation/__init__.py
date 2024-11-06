"""
census_transportation init
"""

from .data_process.acs import main as process_acs
from .data_process.travel_time import main as process_travel_time
from .data_process.life_expectancy import main as process_life_expectancy
from .data_process.means_of_transport import main as process_means_of_transport

from .to_tensor.acs_life_expectancy import main as to_tensor_acs_life_expectancy
from .to_tensor.m_of_t_life_expectancy import main as to_tensor_m_of_t_life_expectancy

from .train.nn_acs_life_expectancy import main as train_acs_life_expectancy
from .train.nn_acs_life_expectancy import NeuralNetwork as model_acs_life_expectancy
from .train.nn_m_of_t_life_expectancy import main as train_m_of_t_life_expectancy
from .train.nn_m_of_t_life_expectancy import NeuralNetwork as model_m_of_t_life_expectancy
