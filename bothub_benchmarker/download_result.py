import time
import os
import posixpath
from utils import get_train_job_status, download_folder_structure_from_bucket, connect_to_storage, bothub_bucket
from argparse import ArgumentParser


def download_benchmark_result(job_id, dl_path):
    status = {
        "QUEUED": 1,
        "PREPARING": 1,
        "RUNNING": 1,
        "SUCCEEDED": 2,
        "FAILED": 3,
        "CANCELLING": 3,
        "CANCELLED": 3,
        "STATE_UNSPECIFIED": 3,
    }

    while True:
        job_response = get_train_job_status(job_id)
        job_status = status.get(job_response.get('state'))
        if job_status == 2:
            print('job is done, downloading results..')
            download_folder_structure_from_bucket(connect_to_storage(bothub_bucket), dl_path, posixpath.join('results', job_id))
            return
        if job_status == 3:
            print('job failed')
            print(job_response.get('errorMessage'))
            return
        if job_status == 1:
            print('job in progress, checking again in 5 seconds')
            time.sleep(5)
            continue


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-id", "--job-id", dest="job_id",
                        help="Job id to look for", required=True)
    parser.add_argument("-out", "--output", dest="output", default="downloaded_results",
                        help="Downloading directory path")

    args = vars(parser.parse_args())

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bothub-273521-b2134fc6b1df.json"
    job_id = args.get('job_id')
    dl_path = args.get('output')

    download_benchmark_result(job_id, posixpath.join(dl_path, job_id))
