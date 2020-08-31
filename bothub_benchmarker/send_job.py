import os
import logging
import posixpath
from googleapiclient import discovery
from googleapiclient import errors
from utils import upload_folder_to_bucket, connect_to_storage
from argparse import ArgumentParser


def send_job_train_ai_platform(job_id, configs_path, datasets_dir, use_spacy=False, use_tensorboard=False, false_positive_benchmark=False):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bothub-273521-b2134fc6b1df.json"

    bucket = connect_to_storage('bothub_benchmark')

    upload_folder_to_bucket(bucket, configs_path, posixpath.join('data', job_id, 'configs'), recursive_upload=False)
    upload_folder_to_bucket(bucket, datasets_dir, posixpath.join('data', job_id, 'data_to_evaluate'), recursive_upload=False)

    package_uris = ["gs://bothub_benchmark/bothub_benchmarker-1.0.0.tar.gz"]
    if use_spacy:
        package_uris.append("gs://bothub_benchmark/pt_nilc_word2vec_cbow_600-1.0.0.zip")

    training_inputs = {
        "scaleTier": "CUSTOM",
        "masterType": "standard_p100",
        "package_uris": package_uris,
        "pythonModule": "bothub_benchmarker.ai_plataform_benchmark",
        "args": [
            '--job-id',
            job_id,
            '--use-tensorboard',
            str(use_tensorboard),
            '--false-positive-benchmark',
            str(false_positive_benchmark)
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
    print(f'{job_id} benchmark job sent')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-id", "--job-id", dest="job_id",
                        help="Unique job id", required=True)
    parser.add_argument("-b", "--benchmark", dest="benchmark_method",
                        help="crossvalidation [cv] | tensorboard [tb] | falsepositive [fp]", required=True)
    parser.add_argument("-s", "--spacy", dest="use_spacy",
                        help="Load spacy", action='store_true')
    parser.set_defaults(use_spacy=False)

    args = vars(parser.parse_args())

    args_id = args.get("job_id")
    args_benchmark_method = args.get("benchmark_method")
    args_spacy = args.get("use_spacy")

    # Default cross-validation
    args_use_tensorboard = False
    args_false_positive_benchmark = False
    data_path = "data_to_evaluate"

    if args_benchmark_method == "tensorboard" or args_benchmark_method == "tb":
        args_use_tensorboard = True
    elif args_benchmark_method == "falsepositive" or args_benchmark_method == "fp":
        args_false_positive_benchmark = True
        data_path = "false_positive_data"

    send_job_train_ai_platform(args_id,
                               posixpath.join('benchmark_sources', 'configs'),
                               posixpath.join('benchmark_sources', data_path),
                               use_spacy=args_spacy, use_tensorboard=args_use_tensorboard,
                               false_positive_benchmark=args_false_positive_benchmark)
