import time
import os
import logging
from .utils import get_train_job_status, download_bucket_folder
from google.cloud import storage

log = logging.getLogger("benchmark-log")


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
        print(job_response)
        job_status = status.get(job_response.get('state'))
        if job_status == 2:
            bucket_name = 'bothub_benchmark'
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name)
            log.info('job is done, downloading results..')
            download_bucket_folder(bucket, os.path.join(job_id, 'results'), dl_path)
            log.info('results downloaded')
            return
        if job_status == 3:
            log.info('job failed')
            # log.error(job_response.get('error')) what is the key with the error?
            return
        if job_status == 1:
            log.info('job in progress')
            time.sleep(5)
            continue


if __name__ == '__main__':
    job_id = 'benchmark_test'
    dl_path = 'bothub_benchmarker/benchmark_output'
    download_benchmark_result(job_id, os.path.join(dl_path, job_id))

