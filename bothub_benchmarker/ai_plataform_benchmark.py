import argparse
import os
import posixpath
from bothub_benchmarker.benchmark import benchmark, tensorboard_benchmark
from bothub_benchmarker.false_positive_benchmark import false_positive_benchmark
from bothub_benchmarker.utils import download_bucket_folder, upload_folder_to_bucket, connect_to_storage


def spacy_setup():
    os.system("python3 -m spacy link pt_nilc_word2vec_cbow_600 pt")


def ai_plataform():
    spacy_setup()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-id',
        help='Job identification')
    parser.add_argument(
        '--use-tensorboard',
        default=False,
        help='If true will not use cross validation')
    parser.add_argument(
        '--false-positive-benchmark',
        default=False,
        help='If true will not use cross validation')

    arguments, _ = parser.parse_known_args()

    bucket = connect_to_storage('bothub_benchmark')

    configs_dir = 'benchmark_sources/configs/'
    data_dir = 'benchmark_sources/data_to_evaluate/'
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    download_bucket_folder(bucket, configs_dir, posixpath.join('data', arguments.job_id, 'configs'))
    download_bucket_folder(bucket, data_dir, posixpath.join('data', arguments.job_id, 'data_to_evaluate'))

    if arguments.use_tensorboard == "True":
        tensorboard_benchmark(arguments.job_id, configs_dir, data_dir)
    elif arguments.false_positive_benchmark == "True":
        false_positive_benchmark(arguments.job_id, configs_dir, data_dir, bucket)
    else:
        benchmark(arguments.job_id, configs_dir, data_dir, bucket)


if __name__ == '__main__':
    ai_plataform()
