import os
import sys
import resource
import platform
# from evaluate_new import benchmark

model_name = 'bot_data_ed654d7d-d752-4df6-9e80-c26c2887e183'


def rasa_train_batch():
    datasets = os.listdir('../benchmark_sources/data_to_evaluate/')
    print(datasets)
    for dataset in datasets:
        command = 'python -m rasa train nlu'
        command += ' -u ../benchmark_sources/data_to_evaluate/' + dataset
        command += ' --fixed-model-name ' + dataset
        command += ' -c ../benchmark_sources/configs/bothub_config_tensorflow_analyze_char.yml'
        print(command)
        os.system(command)


def rasa_train():
    dataset = 'nina_train.md'
    command = 'python -m rasa train nlu'
    # command += ' --help'
    command += ' -u ../benchmark_sources/data_to_evaluate/' + dataset
    command += ' --fixed-model-name ' + dataset
    command += ' -c ../benchmark_sources/configs/bothub_config_tensorflow_analyze_char.yml'
    print(command)
    os.system(command)


def rasa_test():
    model_name = 'nina_train.md'
    test_dataset = 'nina_unfpa.md'
    command = 'python -m rasa test nlu'
    # command += ' --help'
    command += ' -m models/' + model_name + '.tar.gz'
    command += ' -u ../benchmark_sources/data_to_evaluate/' + test_dataset
    print(command)
    os.system(command)


def rasa_shell():
    command = 'python -m rasa shell nlu'
    # command += ' --help'
    command += ' -m models/' + model_name + '.tar.gz'
    print(command)
    os.system(command)


def main():
    # rasa_train()
    # rasa_test()
    rasa_shell()


main()
