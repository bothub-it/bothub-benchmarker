from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
import subprocess
import os


def rasa_train():
    training_data = load_data('../data/training_data.json')
    trainer = Trainer(config.load("nlu_config.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('models')


def rasa_evaluate():
    command = ''
    command += 'python -m rasa_nlu.evaluate'
    command += ' --data ../data/test_data.json'
    command += ' --model models/default/model_20191015-173038'
    command += ' --config nlu_config.yml'
    command += ' --report report'
    command += ' --successes successes'
    command += ' --errors errors'
    command += ' --histogram histogram'
    command += ' --confmat confmat'
    # command += ' --mode crossvalidation'
    # command += ' --folds 2'
    print(command)
    os.system(command)


def rasa_evaluate_cross_val():
    command = ''
    command += 'python -m rasa_nlu.evaluate'
    command += ' --data ../data/AskUbuntuCorpus_rasa.json'
    command += ' --config nlu_config.yml'
    command += ' --report report'
    command += ' --successes successes'
    command += ' --errors errors'
    command += ' --histogram histogram'
    command += ' --confmat confmat'
    command += ' --mode crossvalidation'
    command += ' --folds 2'
    print(command)
    os.system(command)


# rasa_train()
rasa_evaluate()
