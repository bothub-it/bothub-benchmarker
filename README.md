# Weni Developer Benchmark module

---

This module provides evaluation tools running locally or through Jobs on Google AI Platform


## How to use

There are two types of benchmarks:

- cross-validation 
  - performs a 3-fold cross-validation
- tensorboard
  - performs a validation with 25% of dataset and creates tensorboard logs

### AI-Platform

---
Run ```python send_job.py -h``` for details

It will show the available parameters needed to set your job ID and benchmark type

You need to set the env. variable ```GOOGLE_APPLICATION_CREDENTIALS``` the path of .json file containing credentials

Your benchmark_source files will be uploaded to the google bucket:
- config.yml files from folder /configs
- datasets.md files from folder /data_to_evaluate

Each config will be run against each dataset, resulting in (n_configs * n_datasets) different benchmarks

After the job is finished, you can download the results from bucket using ```python download_result.py -id <JOB_ID> -out <OUTPUT_PATH>```


### Local

---
You can run directly the functions inside benchmark.py to perform local benchmarks

Each config from folder /configs will be run against each dataset in folder /data_to_evaluate, resulting in (n_configs * n_datasets) different benchmarks

## How to develop

---

Update setup.py version

run ```python setup.py sdist bdist_wheel``` to generate .tar.gz wheel

Upload .tar.gz file to google cloud benchmark bucket

your ```package_uris``` from send_job.py should be pointing to the new file
