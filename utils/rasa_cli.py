import os
# from evaluate_new import benchmark

config_name = 'diet.yml'
dataset = 'nina_train.md'


def rasa_train_batch():
    datasets = os.listdir('../benchmark_sources/data_to_evaluate/')
    print(datasets)
    for dataset in datasets:
        command = 'python -m rasa train nlu'
        command += ' -u ../benchmark_sources/data_to_evaluate/' + dataset
        command += ' --fixed-model-name ' + dataset
        command += ' -c ../benchmark_sources/configs/' + config_name
        print(command)
        os.system(command)


def rasa_train():
    command = 'python -m rasa train nlu'
    # command += ' --help'
    command += ' -u ../benchmark_sources/data_to_evaluate/' + dataset
    command += ' --fixed-model-name ' + dataset
    command += ' -c ../benchmark_sources/configs/' + config_name
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
    command += ' -m models/' + dataset + '.tar.gz'
    print(command)
    os.system(command)


def main():
    # rasa_train()
    # rasa_test()
    rasa_shell()


main()
