import subprocess
import os


def rasa_train_model():
    command = ''
    command += 'python -m rasa train nlu'
    command += ' -u ../data_to_evaluate/training_data.json'
    command += ' --config config.yml'
    command += ' --out models'
    print(command)
    os.system(command)


def rasa_split_data():
    command = ''
    command += 'python -m rasa data_to_evaluate split nlu'
    command += ' -u ../data_to_evaluate/AskUbuntuCorpus_rasa.json'
    command += ' --training-fraction 0.8'
    command += ' --out ../data_to_evaluate'
    print(command)
    os.system(command)


def rasa_evaluate_model():
    command = ''
    command += 'python -m rasa test nlu'
    command += ' -u ../data_to_evaluate/test_data.json'
    command += ' --model models/nlu-20191015-171930.tar.gz'
    command += ' --out benchmark'
    print(command)
    os.system(command)


def rasa_evaluate_cross_val():
    command = ''
    command += 'python -m rasa test nlu'
    command += ' -u ../data_to_evaluate/ask_ubuntu_test_data.json'
    command += ' --config config.yml'
    command += ' --out benchmark'
    command += ' --cross-validation'
    command += ' --folds 2'
    print(command)
    os.system(command)


# rasa_split_data()
# rasa_train_model()
# rasa_evaluate_model()
rasa_evaluate_cross_val()
