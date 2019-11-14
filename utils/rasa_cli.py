import os
import sys
import resource
import platform
# from evaluate_new import benchmark

model_name = 'strip_acent'


def rasa_train():
    command = 'python -m rasa train nlu'
    # command += ' --help'
    command += ' -u benchmark_sources/data_to_evaluate/sac_viario_sup_test_DELETAR_DEPOIS.md'
    command += ' --fixed-model-name ' + model_name
    command += ' -c benchmark_sources/configs/config_dump/bothub_config_tensorflow_analyze_char_stip_accent.yml'
    print(command)
    os.system(command)


def rasa_shell():
    command = 'python -m rasa shell nlu'
    # command += ' --help'
    command += ' -m models/' + model_name + '.tar.gz'
    print(command)
    os.system(command)


def main():
    rasa_train()
    rasa_shell()


main()
