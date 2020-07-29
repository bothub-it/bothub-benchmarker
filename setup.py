from setuptools import setup, find_packages

setup(
    name='bothub_benchmarker',
    version='1.0.0',
    description='Bothub Benchmark Tool',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'rasa==1.10.8',
        'transformers==2.11.0',
        'google-api-python-client==1.8.3',
        'spacy>=2.1,<2.2',
        'google-cloud-storage==1.29.0',
        'unidecode==1.1.1',
        'torch',
        'torchvision',
        'unidecode'
    ],
)