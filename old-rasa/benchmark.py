from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
import subprocess
import os

model_name = 'rasa_model'


def rasa_train():
    command = 'python -m rasa_nlu.train'
    command += ' --data ../data/ask_ubuntu_training_data.json'
    command += ' -o models'
    command += ' --fixed_model_name ' + model_name
    command += ' --config nlu_config.yml'

    print(command)
    os.system(command)


def rasa_evaluate():
    command = 'python evaluate.py'
    command += ' --data ../data/ask_ubuntu_test_data.json'
    command += ' --model models/default/' + model_name
    command += ' --config nlu_config.yml'
    command += ' --report benchmark/report'
    command += ' --successes benchmark/successes'
    command += ' --errors benchmark/errors'
    command += ' --histogram benchmark/histogram'
    command += ' --confmat benchmark/confmat'
    # command += ' --mode crossvalidation'
    command += ' --folds 2'
    print(command)
    os.system(command)


def rasa_evaluate_cross_val():
    command = 'python evaluate.py'
    command += ' --data ../data/WebApplicationsCorpus_rasa.json'
    command += ' --config nlu_config.yml'
    # command += ' --report benchmark/report'
    # command += ' --successes benchmark/successes'
    command += ' --errors errors'
    # command += ' --histogram histogram'
    # command += ' --confmat confmat'
    command += ' --mode crossvalidation'
    command += ' --folds 2'
    print(command)
    os.system(command)


# rasa_train()
rasa_evaluate()
# rasa_evaluate_cross_val()
