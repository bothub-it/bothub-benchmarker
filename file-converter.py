import json
from rasa.core.validator import Validator
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


class FileConverter:
    def __init__(self, path, filename):
        self.path = path
        self.file = filename
        self.filename = filename.split('.')[0]
        self.extension = filename.split('.')[1]
        self.content = open(path + filename, 'r').read()

    def nlu_corpora_to_json(self):
        data = json.loads(self.content)
        rasa_json = {
            "rasa_nlu_data": {
                "common_examples": [],
                "regex_features": [],
                "entity_synonyms": []
            }
        }
        size = len(data['sentences'])
        for i in range(size):
            entities = []
            for entity in data['sentences'][i]['entities']:
                out_entity = {'start': entity['start'],
                              'end': entity['stop'],
                              'value': entity['text'],
                              'entity': entity['entity']}
                entities.append(out_entity)

            item = {'text': data['sentences'][i]['text'],
                    'intent': data['sentences'][i]['intent'],
                    'entities': entities}

            rasa_json['rasa_nlu_data']['common_examples'].append(item)
        open('data/' + self.filename + '_rasa.' + self.extension, 'w').write(json.dumps(rasa_json))


files_to_convert = ['AskUbuntuCorpus.json', 'ChatbotCorpus.json', 'WebApplicationsCorpus.json']
for file in files_to_convert:
    a = FileConverter('git_data/', file)
    print(file)
    a.nlu_corpora_to_json()
