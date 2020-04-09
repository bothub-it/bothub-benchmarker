import json
import os
import time
from collections import OrderedDict

import psutil
# from evaluate_new import benchmark
from typing import (
    Iterable,
    Iterator,
    Tuple,
    List,
    Set,
    Optional,
    Text,
    Union,
    Dict,
    Any,
)
from rasa.nlu import utils, config, training_data
from rasa.nlu.model import Trainer
from rasa.nlu.components import Component
from utils.debug_parse import DebugSentenceLime

config_name = 'diet.yml'
dataset = 'nina_train.md'
PRETRAINED_EXTRACTORS = {"DucklingHTTPExtractor", "SpacyEntityExtractor"}


def remove_pretrained_extractors(pipeline: List[Component]) -> List[Component]:
    pipeline = [c for c in pipeline if c.name not in PRETRAINED_EXTRACTORS]
    return pipeline


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

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'


def train_model(config_path, data_path):
    nlu_config = config.load(config_path)
    data = training_data.load_data(data_path)
    try:
        trainer = Trainer(nlu_config)
        trainer.pipeline = remove_pretrained_extractors(trainer.pipeline)
    except OSError:
        raise
    interpreter = trainer.train(data)
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)
    return interpreter


def format_debug_parse_output(result_per_word, r):
    for word in result_per_word:
        result_per_word[word] = sorted(result_per_word[word], key=lambda k: k['relevance'], reverse=True)
    result_per_word = OrderedDict(sorted(result_per_word.items(), key=lambda t:t[1][0]["relevance"], reverse=True))
    out = OrderedDict([("intent", r.get("intent", None)), ("words", result_per_word)])
    return out


def n_samples_by_sentence_lenght(sentence):
    word_count = len(sentence.split(' '))
    return word_count * 1000


def rasa_shell_debug(config_path, data_path):
    interpreter = train_model(config_path, data_path)
    while True:
        print('write a sentence to predict')
        sentence = input()
        print('debug:')
        debug = DebugSentenceLime(interpreter, ['regime_fios_malhas', 'regime_especial', 'regime_brita_gesso', 'baixa_st', 're_tecidos', 'regime_comunicacao', 'beneficio_substituicao_tributaria', 'bias', 'exclusao_regime_edital', 'regime_restaurantes', 'regime_agua', 'incentivo_fiscal' ,'diferimento_icms' ,'credito_acosplanos'])
        r = interpreter.parse(sentence)
        n_samples = n_samples_by_sentence_lenght(sentence)
        print(n_samples)
        result_per_word = debug.get_result_per_word(sentence, n_samples)
        print(json.dumps(format_debug_parse_output(result_per_word, r), indent=2))


def profile_rasa_shell(config_path, data_path):
    load_time = time.time()
    interpreter = train_model(config_path, data_path)
    process = psutil.Process(os.getpid())
    print('usage: ', format_bytes(process.memory_info().rss))  # in bytes
    print('model load time:', time.time() - load_time)
    print('write a sentence to predict')
    while True:
        sentence = input()
        parse_time = time.time()
        print(interpreter.parse(sentence))
        print('parse time: ', time.time() - parse_time)


def main():
    # rasa_train()
    # rasa_test()
    rasa_shell_debug('../benchmark_sources/configs/bothub_config_tensorflow_analyze_char.yml',
                     '../benchmark_sources/data_to_evaluate/iac.md')
    # profile_rasa_shell('../data/configs/diet_conveRT.yml', '../data/ChatbotCorpus_rasa.json')

main()
