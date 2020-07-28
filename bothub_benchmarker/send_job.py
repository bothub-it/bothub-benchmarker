import google.oauth2.credentials
import os
from googleapiclient import discovery
from googleapiclient import errors
from google.cloud import storage
from utils import upload_folder_to_bucket, connect_to_storage


def send_job_train_ai_platform(configs_path, datasets_dir, job_id, use_spacy=False):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bothub-273521-b2134fc6b1df.json"

    bucket = connect_to_storage('bothub_benchmark')

    upload_folder_to_bucket(bucket, configs_path, os.path.join(job_id, 'configs'))
    upload_folder_to_bucket(bucket, datasets_dir, os.path.join(job_id, 'data_to_evaluate'))

    packageUris = ["gs://bothub_benchmark/bothub_benchmarker-1.0.0.tar.gz"]
    if use_spacy:
        packageUris.append("gs://bothub_benchmark/pt_nilc_word2vec_cbow_600-1.0.0.zip")
    training_inputs = {
        "scaleTier": "BASIC_GPU",
        "packageUris": packageUris,
        "pythonModule": "bothub_benchmarker.ai_plataform_benchmark",
        "args": [
            '--job-id',
            job_id,
        ],
        "region": "us-east1",
        "jobDir": "gs://bothub_benchmark",
        "runtimeVersion": "2.1",
        'pythonVersion': '3.7'
    }

    job_spec = {"jobId": job_id, "trainingInput": training_inputs}

    project_id = "projects/bothub-273521"

    # Consiga uma representação em Python dos serviços do AI Platform Training:
    cloudml = discovery.build("ml", "v1")

    # Crie e envie sua solicitação:
    request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)

    try:
        request.execute()
    except errors.HttpError as err:
        raise Exception(err)


if __name__ == '__main__':
    send_job_train_ai_platform('benchmark_sources/configs',
                               'benchmark_sources/data_to_evaluate',
                               'benchmark_download_results_test_1')
