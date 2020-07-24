import argparse
import os
import glob
from bothub_benchmarker.benchmark import benchmark
from google.cloud import storage


def download_bucket_folder(bucket, bucket_dir, dl_dir):
    blobs = bucket.list_blobs(prefix=bucket_dir)  # Get list of files
    for blob in blobs:
        blob_name = blob.name
        print(blob.name)
        dst_file_name = blob_name.replace(bucket_dir, '')
        if '/' in dst_file_name or '.' not in dst_file_name:
            continue
        blob.download_to_filename(os.path.join(dl_dir, dst_file_name))


def upload_local_directory_to_gcs(bucket, local_path, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
           upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
           remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
           blob = bucket.blob(remote_path)
           blob.upload_from_filename(local_file)


def ai_plataform():
    parser = argparse.ArgumentParser()
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bothub-273521-b2134fc6b1df.json"
    parser.add_argument(
        '--configs-bucket-dir',
        help='Bucket folder where the pipelines to be evaluate are in')
    parser.add_argument(
        '--datasets-bucket-dir',
        help='Bucket folder where the datasets to be evaluate are in')
    parser.add_argument(
        '--results-dir',
        help='Bucket folder where the benchmark results will be saved')

    arguments, _ = parser.parse_known_args()

    bucket_name = 'bothub_benchmark'

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    configs_dir = 'benchmark_sources/configs/'
    data_dir = 'benchmark_sources/data_to_evaluate/'
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    download_bucket_folder(bucket, arguments.configs_bucket_dir, configs_dir)
    download_bucket_folder(bucket, arguments.datasets_bucket_dir, data_dir)

    benchmark(arguments.results_dir, configs_dir, data_dir)

    upload_local_directory_to_gcs(bucket, arguments.results_dir, arguments.results_dir)
    # download_bucket_folder(bucket, configs_dir, configs_dir)
    # download_bucket_folder(bucket, data_dir, data_dir)
    #
    # benchmark('hello_world_test', configs_dir, data_dir)


if __name__ == '__main__':
    ai_plataform()
