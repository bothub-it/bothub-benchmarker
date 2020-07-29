import os
import glob
import logging
import posixpath
from googleapiclient import discovery
from google.cloud import storage

bothub_bucket = 'bothub_benchmark'


def upload_folder_to_bucket(bucket, local_path, bucket_path, recursive_upload=True):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file) and recursive_upload:
            upload_folder_to_bucket(bucket, local_file, bucket_path + "/" + os.path.basename(local_file))
        else:
            if '.' not in local_file:
                continue
            remote_path = posixpath.join(bucket_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            print('sent {}'.format(local_file.split("/")[-1]))
            blob.upload_from_filename(local_file)


def find_occurrences(s, ch):  # to find position of '/' in blob path ,used to create folders in local storage
    return [i for i, letter in enumerate(s) if letter == ch]


def download_folder_structure_from_bucket(bucket, local_path, bucket_path):

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    blobs = list(bucket.list_blobs(prefix=bucket_path))

    for blob in blobs:
        start_loc = 0
        folder_loc = find_occurrences(blob.name.replace(bucket_path, ''), '/')
        if not blob.name.endswith("/"):
            if blob.name.replace(bucket_path, '').find("/") == -1:
                download_path = local_path + '/' + blob.name.replace(bucket_path, '')
                blob.download_to_filename(download_path)
            else:
                for folder in folder_loc:
                    if not os.path.exists(local_path + '/' + blob.name.replace(bucket_path, '')[start_loc:folder]):
                        create_folder = local_path + '/' + blob.name.replace(bucket_path, '')[0:start_loc] + '/' + blob.name.replace(bucket_path, '')[start_loc:folder]
                        start_loc = folder + 1
                        os.makedirs(create_folder)

                download_path = local_path + '/' + blob.name.replace(bucket_path, '')

                blob.download_to_filename(download_path)
                print(blob.name.replace(bucket_path, '')[0:blob.name.replace(bucket_path, '').find("/")])

    print('blob {} downloaded to {}.'.format(bucket_path, local_path))


def download_bucket_folder(bucket, local_path, bucket_path):
    blobs = bucket.list_blobs(prefix=bucket_path)  # Get list of files

    for blob in blobs:
        blob_name = blob.name
        dst_file_name = blob_name.replace(bucket_path, '')
        if dst_file_name[0] == '/':
            dst_file_name = dst_file_name[1:]
        if '/' in dst_file_name or '.' not in dst_file_name:
            continue
        blob.download_to_filename(posixpath.join(local_path, dst_file_name))

    print('blob {} downloaded to {}.'.format(bucket_path, local_path))


def get_train_job_status(job_id):
    job_name = f"projects/bothub-273521/jobs/{job_id}"

    cloudml = discovery.build( "ml", "v1")

    request = cloudml.projects().jobs().get(name=job_name)

    try:
        response = request.execute()
    except Exception as e:
        raise Exception(e)

    if response is None:
        raise Exception("Got None as response.")

    return response


def connect_to_storage(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

