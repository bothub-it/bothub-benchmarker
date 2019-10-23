import os
import sys

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
    out_directory = 'benchmark'
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    command = 'python evaluate.py'
    command += ' --data ../data/ask_ubuntu_test_data.json'
    command += ' --model models/default/' + model_name
    command += ' --report ' + out_directory + '/report'
    command += ' --successes ' + out_directory + '/successes'
    command += ' --errors ' + out_directory + '/errors'
    command += ' --histogram ' + out_directory + '/histogram'
    command += ' --confmat ' + out_directory + '/confmat'
    print(command)
    os.system(command)


def rasa_benchmark():
    out_directory = 'benchmark-3'
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    command = 'python evaluate.py'
    command += ' --data ../data/AskUbuntuCorpus_rasa.json'
    command += ' --config nlu_config.yml'
    command += ' --report ' + out_directory + '/report'
    command += ' --successes ' + out_directory + '/successes'
    command += ' --errors ' + out_directory + '/errors'
    command += ' --histogram ' + out_directory + '/histogram'
    command += ' --confmat ' + out_directory + '/confmat'
    command += ' --mode benchmark'
    command += ' --folds 5'
    print(command)
    os.system(command)


def rasa_evaluate_cross_val():
    out_directory = 'benchmark'
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    command = 'python evaluate.py'
    command += ' --data ../data/WebApplicationsCorpus_rasa.json'
    command += ' --config nlu_config.yml'
    command += ' --report ' + out_directory + '/report'
    command += ' --successes ' + out_directory + '/successes'
    command += ' --errors ' + out_directory + '/errors'
    command += ' --histogram ' + out_directory + '/histogram'
    command += ' --confmat ' + out_directory + '/confmat'
    command += ' --mode crossvalidation'
    command += ' --folds 5'
    print(command)
    os.system(command)


def main():
    print(' 1 - train  2 - evaluate  3 - benchmark  4 - cross validation  5 - train + evaluate')
    x = input()
    x = int(x)
    x = 3
    if x == 1:
        rasa_train()
    elif x == 2:
        rasa_evaluate()
    elif x == 3:
        rasa_benchmark()
    elif x == 4:
        rasa_evaluate_cross_val()
    elif x == 5:
        rasa_train()
        rasa_evaluate()


main()
