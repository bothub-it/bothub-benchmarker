import argparse
import google.oauth2.credentials
import os
import glob
from bothub_benchmarker.benchmark import benchmark
from google.cloud import storage
from googleapiclient import discovery
from googleapiclient import errors


def send_job_train_ai_platform(use_spacy=False):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bothub-273521-b2134fc6b1df.json"
    packageUris = ["gs://bothub_benchmark/bothub_benchmarker-1.0.0.tar.gz"]
    if use_spacy:
        packageUris.append("gs://bothub_benchmark/pt_nilc_word2vec_cbow_600-1.0.0.zip")
    training_inputs = {
        "scaleTier": "BASIC_GPU",
        "packageUris": packageUris,
        "pythonModule": "bothub_benchmarker.ai_plataform_benchmark",
        "args": [
            '--configs-bucket-dir',
            'benchmark_sources/configs/',
            '--datasets-bucket-dir',
            'benchmark_sources/data_to_evaluate/',
            '--out-bucket-dir',
            'hello_world_benchmark'
        ],
        "region": "us-east1",
        "jobDir": "gs://bothub_benchmark",
        "runtimeVersion": "2.1",
        'pythonVersion': '3.7'
    }

    job_spec = {"jobId": "hello_world_benchmark_try_15", "trainingInput": training_inputs}

    project_id = "projects/bothub-273521"

    credentials = google.oauth2.credentials.Credentials(
        "access_token",
        refresh_token="1//04b6e_XLs_vBxCgYIARAAGAQSNwF-L9IruPZU5hMQMCcFczeZCqRKUmkqxYGIcDC_75PHcgnMzgb15nUqpXeVjyqsuEhM6xU-rxs",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="615988211650-71d28ugemkdi3cas70am686akeif3mq8.apps.googleusercontent.com",
        client_secret="6LhBP7kaAosWZM0Xv4Jh1vBr",
    )

    # Consiga uma representação em Python dos serviços do AI Platform Training:
    cloudml = discovery.build(
        "ml", "v1", credentials=credentials, cache_discovery=False
    )

    # Crie e envie sua solicitação:
    request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)

    try:
        request.execute()
    except errors.HttpError as err:
        raise Exception()


def download_bucket_folder(bucket, bucket_dir, dl_dir):
    blobs = bucket.list_blobs(prefix=bucket_dir)  # Get list of files
    # print(bucket_dir)
    for blob in blobs:
        blob_name = blob.name
        print(blob.name)
        dst_file_name = blob_name.replace(bucket_dir, '')
        if '/' in dst_file_name or '.' not in dst_file_name:
            continue
        # extract the final directory and create it in the destination path if it does not exist
        # dl_dir = dst_file_name.replace('/' + os.path.basename(dst_file_name), '')
        # download the blob object
        # print(dl_dir)
        # print('path', os.path.join(dl_dir, dst_file_name))
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
        '--out-bucket-dir',
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

    benchmark(arguments.out_bucket_dir, configs_dir, data_dir)

    # download_bucket_folder(bucket, configs_dir, configs_dir)
    # download_bucket_folder(bucket, data_dir, data_dir)
    #
    # benchmark('hello_world_test', configs_dir, data_dir)


if __name__ == '__main__':
    # ai_plataform()
    send_job_train_ai_platform()
