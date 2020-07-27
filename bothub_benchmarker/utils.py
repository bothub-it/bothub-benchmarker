import os
import glob

def upload_folder_to_bucket(bucket, local_path, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_folder_to_bucket(bucket, local_file, gcs_path + "/" + os.path.basename(local_file))
        else:
           remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
           blob = bucket.blob(remote_path)
           blob.upload_from_filename(local_file)


def download_bucket_folder(bucket, bucket_dir, dl_dir):
    blobs = bucket.list_blobs(prefix=bucket_dir)  # Get list of files
    for blob in blobs:
        blob_name = blob.name
        dst_file_name = blob_name.replace(bucket_dir, '')
        if '/' in dst_file_name or '.' not in dst_file_name:
            continue
        blob.download_to_filename(os.path.join(dl_dir, dst_file_name))

