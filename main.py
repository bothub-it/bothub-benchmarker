import os
import sys

model_name = 'rasa_model'

def rasa_train():
    command = 'python -m rasa_nlu.train'
    command += ' --data_to_evaluate ../data_to_evaluate/ask_ubuntu_training_data.json'
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
    command += ' --data data/ask_ubuntu_test_data.json'
    command += ' --model models/default/' + model_name
    command += ' --report ' + out_directory + '/report'
    command += ' --successes ' + out_directory + '/successes'
    command += ' --errors ' + out_directory + '/errors'
    command += ' --histogram ' + out_directory + '/histogram'
    command += ' --confmat ' + out_directory + '/confmat'
    print(command)
    os.system(command)


def rasa_benchmark():
    command = 'python evaluate_new.py'
    command += ' --data benchmark_sources/data_to_evaluate'
    command += ' --config benchmark_sources/configs'
    command += ' --mode benchmark'
    command += ' --folds 3'
    print(command)
    os.system(command)


def rasa_evaluate_cross_val():
    out_directory = 'benchmark'
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    command = 'python evaluate_new.py'
    command += ' --data data/WebApplicationsCorpus_rasa.json'
    command += ' --config benchmark_sources/configs/nlu_config.yml'
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
    pass


main()
