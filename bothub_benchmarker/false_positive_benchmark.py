import itertools
import json
import os
import logging
import numpy as np
from collections import defaultdict, namedtuple
from tqdm import tqdm
from typing import (
    Iterable,
    Collection,
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

import rasa.utils.io as io_utils

from rasa.constants import TEST_DATA_FILE, TRAIN_DATA_FILE, NLG_DATA_FILE
from rasa.nlu.constants import (
    DEFAULT_OPEN_UTTERANCE_TYPE,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    OPEN_UTTERANCE_PREDICTION_KEY,
    EXTRACTOR,
    PRETRAINED_EXTRACTORS,
    NO_ENTITY_TAG,
)
from rasa.model import get_model
from rasa.nlu import config, training_data, utils
from rasa.nlu.test import remove_pretrained_extractors, get_eval_data
from rasa.nlu.utils import write_to_file
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer, TrainingData
from rasa.nlu.components import Component
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.utils.tensorflow.constants import ENTITY_RECOGNITION


def get_false_positive_data(interpreter, test_dataset_path):
    result = {}
    with open(test_dataset_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        count = 0
        for line in lines:
            if line[0] == '-':
                print(line)
                # print(json.dumps(interpreter.parse(line), indent=2))
            count += 1
            if count >= 6:
                break
    return result


def false_positive_benchmark(out_directory, config_directory, dataset_directory):
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    else:
        count = 0
        out_directory_temp = out_directory
        while os.path.exists(out_directory_temp):
            out_directory_temp = out_directory + str(count)
            count += 1

    config_size = len(os.listdir(config_directory))
    count_config = 0
    for config_filename in os.listdir(config_directory):
        count_config += 1
        print('######################################')
        print('CURRENT CONFIG :', config_filename, ' PROGRESS:', count_config, '/', config_size)
        print('######################################')
        if config_filename.endswith(".yml"):
            config_path = os.path.join(config_directory, config_filename)
            config_name = config_filename.split('.')[0]
            out_config_directory = out_directory + config_name + '/'
            if not os.path.exists(out_config_directory):
                os.mkdir(out_config_directory)
            datasets_dir_out = 'Datasets_Results/'
            if not os.path.exists(out_config_directory + datasets_dir_out):
                os.mkdir(out_config_directory + datasets_dir_out)
            nlu_config = config.load(config_path)
            try:
                trainer = Trainer(nlu_config)
                trainer.pipeline = remove_pretrained_extractors(trainer.pipeline)
            except OSError:
                raise
            datasets_results = []
            datasets_names = []
            for dataset_filename in os.listdir(dataset_directory):
                dataset_path = os.path.join(dataset_directory, dataset_filename)
                dataset_name = dataset_filename.split('.')
                dataset_name = dataset_name[0]
                if dataset_filename.endswith(".json") or dataset_filename.endswith(".md") and dataset_name.split("_")[-1] != 'test':
                    test_dataset_filename = dataset_name + '_test.' + dataset_filename.split('.')[-1]
                    print(test_dataset_filename)
                    test_dataset_path = os.path.join(dataset_directory, test_dataset_filename)
                    data = training_data.load_data(dataset_path)

                    interpreter = trainer.train(data)
                    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

                    intent_results = get_false_positive_data(
                        interpreter, test_dataset_path
                    )
                    # utils.write_json_to_file('new_result_test', cross_val_results)

                    utils.write_json_to_file(out_config_directory + datasets_dir_out + dataset_name + '_Benchmark',
                                             intent_results)
                    datasets_results.append(intent_results)
                    datasets_names.append(dataset_filename)
            # save_result_by_group(datasets_results, n_folds, out_config_directory, datasets_names)