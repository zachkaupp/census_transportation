"""
main.py
zachkaupp@gmail.com
"""

from .data_process.acs import main as process_acs
from .data_process.travel_time import main as process_travel_time

def main():
    """main()"""
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

if __name__ == "__main__":
    main()
