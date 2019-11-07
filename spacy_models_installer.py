import os

models = 'spacy_models'
for model in os.listdir(models):
    if model.endswith('.zip'):
        pip_install_cmd = 'pip install ' + models + '/' + model
        spacy_model = model.split('-')[0]
        spacy_model = spacy_model.replace('-', '_')
        link_cmd = 'python -m spacy link ' + spacy_model + ' ' + spacy_model
        os.system(pip_install_cmd)
        os.system(link_cmd)