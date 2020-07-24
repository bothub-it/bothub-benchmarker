import google.oauth2.credentials
import os
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
            '--results-dir',
            'hello_world_benchmark'
        ],
        "region": "us-east1",
        "jobDir": "gs://bothub_benchmark",
        "runtimeVersion": "2.1",
        'pythonVersion': '3.7'
    }

    job_spec = {"jobId": "hello_world_benchmark_test_output5", "trainingInput": training_inputs}

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


if __name__ == '__main__':
    send_job_train_ai_platform()
