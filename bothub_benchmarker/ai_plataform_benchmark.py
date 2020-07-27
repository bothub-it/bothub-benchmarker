import argparse
import os
import glob
from bothub_benchmarker.benchmark import benchmark
from google.cloud import storage


def spacy_setup():
    os.system("python -m spacy link pt_nilc_word2vec_cbow_600 pt")


def download_bucket_folder(bucket, bucket_dir, dl_dir):
    blobs = bucket.list_blobs(prefix=bucket_dir)  # Get list of files
    for blob in blobs:
        blob_name = blob.name
        dst_file_name = blob_name.replace(bucket_dir, '')
        if '/' in dst_file_name or '.' not in dst_file_name:
            continue
        blob.download_to_filename(os.path.join(dl_dir, dst_file_name))


def upload_folder_to_bucket(bucket, local_path, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_folder_to_bucket(bucket, local_file, gcs_path + "/" + os.path.basename(local_file))
        else:
           remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
           blob = bucket.blob(remote_path)
           blob.upload_from_filename(local_file)


def ai_plataform():
    spacy_setup()
    parser = argparse.ArgumentParser()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bothub-273521-b2134fc6b1df.json"
    parser.add_argument(
        '--job-id',
        help='Job identification',
        type=int)

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

    upload_folder_to_bucket(bucket, arguments.job_id, arguments.job_id)

    # download_bucket_folder(bucket, configs_dir, configs_dir)
    # download_bucket_folder(bucket, data_dir, data_dir)
    # benchmark('hello_world_test', configs_dir, data_dir)


if __name__ == '__main__':
    ai_plataform()
