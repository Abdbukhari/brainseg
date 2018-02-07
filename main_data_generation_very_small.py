__author__ = 'adeb'

from spynet.utils.utilities import load_config
from data_brain_parcellation import generate_and_save


if __name__ == '__main__':
    """
    Script to create the testing/training datasets given configuration files
    """

    ### Load the config file
    data_cf = load_config("cfg_testing_data_creation_very_small.py")

    ### Generate and write on file the dataset
    generate_and_save(data_cf)

    ### Load the config file
    data_cf = load_config("cfg_training_data_creation_very_small.py")

    ### Generate and write on file the dataset
    generate_and_save(data_cf)