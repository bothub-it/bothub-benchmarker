import argparse
import os
from bothub_benchmarker.benchmark import benchmark
from bothub_benchmarker.utils import download_bucket_folder, upload_folder_to_bucket
from google.cloud import storage


def spacy_setup():
    os.system("python -m spacy link pt_nilc_word2vec_cbow_600 pt")


def ai_plataform():
    spacy_setup()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-id',
        help='Job identification')

    arguments, _ = parser.parse_known_args()

    bucket_name = 'bothub_benchmark'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    configs_dir = 'benchmark_sources/configs/'
    data_dir = 'benchmark_sources/data_to_evaluate/'
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    download_bucket_folder(bucket, os.path.join(arguments.job_id, 'configs'), configs_dir)
    download_bucket_folder(bucket, os.path.join(arguments.job_id, 'data_to_evaluate'), data_dir)

    benchmark(arguments.job_id, configs_dir, data_dir)

    upload_folder_to_bucket(bucket, arguments.job_id, os.path.join(arguments.job_id, 'results'))

    # download_bucket_folder(bucket, configs_dir, configs_dir)
    # download_bucket_folder(bucket, data_dir, data_dir)
    # benchmark('hello_world_test', configs_dir, data_dir)


if __name__ == '__main__':
    ai_plataform()
