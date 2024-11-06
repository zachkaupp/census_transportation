"""
main.py
zachkaupp@gmail.com
"""

from .data_process.acs import main as process_acs
from .data_process.travel_time import main as process_travel_time
from .data_process.life_expectancy import main as process_life_expectancy
from .data_process.means_of_transport import main as process_means_of_transport

from .to_tensor.acs_life_expectancy import main as to_tensor_acs_life_expectancy
from .to_tensor.m_of_t_life_expectancy import main as to_tensor_m_of_t_life_expectancy

from .train.nn_acs_life_expectancy import main as train_acs_life_expectancy
from .train.nn_m_of_t_life_expectancy import main as train_m_of_t_life_expectancy


def main():
    """main()"""
    # data_process ------------------------
    print("Cleaning data files...")
    try:
        process_acs()
        print("Cleaned acs_data.csv")
    except Exception as e:
        print("Failed to clean acs_data.csv")
        raise e

    try:
        process_travel_time()
        print("Cleaned travel_time_to_work.csv")
    except Exception as e:
        print("Failed to clean travel_time_to_work.csv")
        raise e

    try:
        process_life_expectancy()
        print("Cleaned life_expectancy.csv")
    except Exception as e:
        print("Failed to clean life_expectancy.csv")
        raise e

    try:
        process_means_of_transport()
        print("Cleaned means_of_transportation_to_work.csv")
    except Exception as e:
        print("Failed to clean means_of_transportation_to_work.csv")
        raise e

    # to_tensor ----------------------------------
    print("Converting dataframes to TensorDataset...")

    try:
        to_tensor_acs_life_expectancy()
        print("Created acs_life_expectancy.pt dataset")
    except Exception as e:
        print("Failed to create life_expectancy.pt dataset")
        raise e

    try:
        to_tensor_m_of_t_life_expectancy()
        print("Created m_of_t_life_expectancy.pt dataset")
    except Exception as e:
        print("Failed to create m_of_t_life_expectancy.pt dataset")
        raise e

    # train ----------------------------------------
    print("Training models...")

    try:
        train_acs_life_expectancy()
        print("Trained nn_life_expectancy.pth")
    except Exception as e:
        print("Failed to train nn_life_expectancy.pth")
        raise e

    try:
        train_m_of_t_life_expectancy()
        print("Trained nn_m_of_t_life_expectancy.pth")
    except Exception as e:
        print("Failed to train nn_m_of_t_life_expectancy.pth")
        raise e

if __name__ == "__main__":
    main()
