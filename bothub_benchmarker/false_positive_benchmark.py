import os
import posixpath
import logging
from rasa.nlu import config, training_data, utils
from rasa.nlu.test import remove_pretrained_extractors
from rasa.nlu.model import Trainer
from bothub_benchmarker.utils import upload_folder_to_bucket


logger = logging.getLogger(__name__)


def init_config_per_dataset(dataset_directory):
    config_per_dataset_result = {}
    for dataset_filename in os.listdir(dataset_directory):
        if dataset_filename.endswith(".md") and 'test' not in dataset_filename:
            dataset_name = dataset_filename.split('.')[0]
            config_per_dataset_result[dataset_name] = []
    return config_per_dataset_result


def save_config_per_dataset(config_per_dataset_result, dataset_directory, out_directory):
    config_per_dataset_path = posixpath.join(out_directory, 'dataset_results')
    os.makedirs(config_per_dataset_path, exist_ok=True)
    for dataset_filename in os.listdir(dataset_directory):
        if dataset_filename.endswith(".md") and 'test' not in dataset_filename:
            dataset_name = dataset_filename.split('.')[0]
            utils.write_json_to_file(posixpath.join(config_per_dataset_path, dataset_name), config_per_dataset_result.get(dataset_name))


def get_false_positive_data(interpreter, test_dataset_path):
    result = {
        'confidence_bellow_70': 0,
        'confidence_bellow_50': 0,
        'confidence_bellow_30': 0,
        'confidence_none': 0
    }
    data = training_data.load_data(test_dataset_path)
    for example in data.training_examples:
        prediction = interpreter.parse(example.text)
        confidence = prediction.get('intent', {}).get('confidence', 0)
        if confidence < 0.30:
            result['confidence_bellow_30'] += 1
        elif confidence < 0.50:
            result['confidence_bellow_50'] += 1
        elif confidence < 0.70:
            result['confidence_bellow_70'] += 1
        elif confidence == 0:
            result['confidence_none'] += 1

    examples_size = len(data.training_examples)
    print(examples_size)
    print(result['confidence_bellow_30'])
    print(result['confidence_bellow_50'])
    print(result['confidence_bellow_70'])
    print(result['confidence_none'])
    result['confidence_bellow_30'] /= examples_size
    result['confidence_bellow_50'] /= examples_size
    result['confidence_bellow_70'] /= examples_size
    result['confidence_none'] /= examples_size
    print('------------------------')
    print(result['confidence_bellow_30'])
    print(result['confidence_bellow_50'])
    print(result['confidence_bellow_70'])
    print(result['confidence_none'])

    return result


def sum_results(datasets_results, config_name):
    benchmark_result = {
        'confidence_bellow_70': 0,
        'confidence_bellow_50': 0,
        'confidence_bellow_30': 0,
        'confidence_none': 0,
    }
    for result in datasets_results:
        benchmark_result['confidence_bellow_70'] += result['confidence_bellow_70']
        benchmark_result['confidence_bellow_50'] += result['confidence_bellow_50']
        benchmark_result['confidence_bellow_30'] += result['confidence_bellow_30']
        benchmark_result['confidence_none'] += result['confidence_none']

    size = len(datasets_results)
    benchmark_result['config_name'] = config_name
    benchmark_result['confidence_bellow_70'] /= size
    benchmark_result['confidence_bellow_50'] /= size
    benchmark_result['confidence_bellow_30'] /= size
    benchmark_result['confidence_none'] /= size
    return benchmark_result


def false_positive_benchmark(out_directory, config_directory, dataset_directory, lookup_tables_dir, bucket=None):
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    else:
        count = 0
        out_directory_temp = out_directory
        while os.path.exists(out_directory_temp):
            out_directory_temp = out_directory + str(count)
            count += 1

    config_per_dataset_result = init_config_per_dataset(dataset_directory)

    config_size = len(os.listdir(config_directory))
    count_config = 0
    config_results = []

    for config_filename in os.listdir(config_directory):
        count_config += 1
        print('######################################')
        print('CURRENT CONFIG :', config_filename, ' PROGRESS:', count_config, '/', config_size)
        print('######################################')
        if config_filename.endswith(".yml"):
            config_path = posixpath.join(config_directory, config_filename)
            config_name = config_filename.split('.')[0]
            out_config_directory = posixpath.join(out_directory, config_name)
            os.makedirs(out_config_directory, exist_ok=True)
            datasets_dir_out = 'datasets_results/'
            os.makedirs(posixpath.join(out_config_directory, datasets_dir_out), exist_ok=True)

            nlu_config = config.load(config_path)
            try:
                trainer = Trainer(nlu_config)
                trainer.pipeline = remove_pretrained_extractors(trainer.pipeline)
            except OSError:
                raise

            datasets_results = []
            count_dataset = 0
            datasets_size = len(os.listdir(dataset_directory))
            for dataset_filename in os.listdir(dataset_directory):
                count_dataset += 1
                dataset_path = posixpath.join(dataset_directory, dataset_filename)
                dataset_name = dataset_filename.split('.')[0]
                logger.info(f'CURRENT CONFIG : {config_filename} PROGRESS: {count_config}/{config_size}')
                logger.info(f'CURRENT DATASET : {dataset_filename} PROGRESS: {count_dataset}/{datasets_size}')
                if (dataset_filename.endswith(".json") or dataset_filename.endswith(".md")) and dataset_name.split("_")[-1] != 'test':
                    test_dataset_filename = dataset_name + '_test.' + dataset_filename.split('.')[-1]
                    print(f"train: {dataset_filename} test: {test_dataset_filename}")
                    print('##############################################')
                    test_dataset_path = posixpath.join(dataset_directory, test_dataset_filename)
                    data = training_data.load_data(dataset_path)

                    interpreter = trainer.train(data)
                    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

                    intent_results = get_false_positive_data(
                        interpreter, test_dataset_path
                    )

                    temp = intent_results
                    temp['config'] = config_name
                    config_per_dataset_result[dataset_name].append(temp)

                    utils.write_json_to_file(posixpath.join(out_config_directory, datasets_dir_out, dataset_name + '_benchmark'),
                                             intent_results)
                    if bucket is not None:
                        upload_folder_to_bucket(bucket, out_config_directory, posixpath.join('results', out_config_directory))
                    datasets_results.append(intent_results)

            config_result = sum_results(datasets_results, config_name)
            config_results.append(config_result)
            utils.write_json_to_file(posixpath.join(out_config_directory, 'result_mean'), config_result)
    save_config_per_dataset(config_per_dataset_result, dataset_directory, out_directory)
    benchmark_result = sorted(config_results, key=lambda k: k['confidence_bellow_70'])
    utils.write_json_to_file(posixpath.join(out_directory, 'benchmark_result'), benchmark_result)
    if bucket is not None:
        upload_folder_to_bucket(bucket, out_directory, posixpath.join('results', out_directory))


if __name__ == '__main__':
    false_positive_benchmark(out_directory='benchmark_output/false_positive_v3',
                             config_directory='benchmark_sources/configs/',
                             dataset_directory='benchmark_sources/false_positive_data')
