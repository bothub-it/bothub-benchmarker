import requests
import json


class BothubRepository:
    def __init__(self, repo_name, repo_uuid, language, auth_token):
        self.repo_name = repo_name
        self.repo_uuid = repo_uuid
        self.api_url = 'https://api.bothub.it/v2/repository/repository-info/{' + repo_uuid + '}/'
        self.language = language
        self.headers = {'Content-Type': 'application/json',
                        'Authorization': auth_token}

    def get_all_results(self, headers, next_call):
        while next_call is not None:
            response = requests.get(next_call, headers=headers)
            response_json = response.json()
            next_call = response_json.get('next')
            yield response_json.get('results', None)

    def format_entity(self, text, entity, start, end):
        '''
        Returns the correct formatted text with the correct entities
        '''
        return text[0:start] + '[' + text[start:end] + '](' + entity + ')' + text[end:]

    def get_train(self):
        print('TRAIN:')
        with open(self.repo_name + '_train', 'w') as nlu_file:
            response = requests.get(self.api_url, headers=self.headers)
            intents = [intent['value'] for intent in response.json()['intents']]
            print('    intents: ', intents)
            for intent in intents:
                nlu_file.write('\n')
                nlu_file.write(f'## intent:{intent}\n')
                next_call = f'https://api.bothub.it/v2/examples/?intent={intent}&limit=20&repository_uuid={self.repo_uuid}'
                results = self.get_all_results(headers=self.headers, next_call=next_call)
                print('        intent: ', intent, '   results: ', results)
                for page in results:
                    for item in page:
                        print('            intent: ', intent, '   item: ', item)
                        if item['language'] == 'pt_br':
                            text = item['text']
                            for entity in item['entities']:
                                entity_text = text[entity['start']:entity['end']]
                                entity_type = entity['entity']
                                text = text.replace(entity_text, f'[{entity_text}]({entity_type})')
                            nlu_file.write(f'- {text}\n')

    def get_test(self):
        with open(self.repo_name + '_test', 'w') as test_file:
            evaluate_url = f'https://api.bothub.it/v2/repository/evaluate/?repository_uuid={self.repo_uuid}'
            results = self.get_all_results(headers=self.headers, next_call=evaluate_url)
            intents = {}
            for page in results:
                for item in page:
                    if item['language'] == self.language:
                        intent = item['intent']
                        if not intent in intents:
                            intents[intent] = []
                        intents[intent].append(item)
            for key in intents:
                test_file.write(f'\n## intent:{key}\n')
                items = intents[key]
                for item in items:
                    text = item['text']
                    for entity in item['entities']:
                        entity_text = text[entity['start']:entity['end']]
                        entity_type = entity['entity']
                        text = text.replace(entity_text, f'[{entity_text}]({entity_type})')
                    test_file.write(f'- {text}\n')

    def get_merged_train_test(self):
        test_count = 0
        train_count = 0
        overlap_count = 0
        with open(self.repo_name + '.md', 'w') as nlu_file:
            evaluate_url = f'https://api.bothub.it/v2/repository/evaluate/?repository_uuid={self.repo_uuid}'
            test_results = self.get_all_results(headers=self.headers, next_call=evaluate_url)
            test_dict = {}

            for page in test_results:
                for item in page:
                    print(item)
                    if item['language'] == self.language:
                        intent = item['intent']
                        if not intent in test_dict:
                            test_dict[intent] = []
                        test_count += 1
                        test_dict[intent].append(item)

            response = requests.get(self.api_url, headers=self.headers)
            print(response.json())
            train_intents = [intent['value'] for intent in response.json()['intents']]
            # print('    train intents: ', train_intents)
            # print('    test intents: ', test_intents)
            for intent in train_intents:
                nlu_file.write('\n')
                nlu_file.write(f'## intent:{intent}\n')
                next_call = f'https://api.bothub.it/v2/repository/examples/?intent={intent}&limit=20&repository_uuid={self.repo_uuid}'
                train_results = self.get_all_results(headers=self.headers, next_call=next_call)
                print('        intent: ', intent, '   results: ', train_results)
                train_items = []
                for page in train_results:
                    for item in page:
                        if item['language'] == 'pt_br':
                            text = item['text']
                            for entity in item['entities']:
                                entity_text = text[entity['start']:entity['end']]
                                entity_type = entity['entity']
                                text = text.replace(entity_text, f'[{entity_text}]({entity_type})')
                            train_items.append(text)
                            train_count += 1
                            print('            intent: ', intent, '   text: ', text)
                            nlu_file.write(f'- {text}\n')

                items = test_dict[intent] if intent in test_dict else {}
                print(train_items)
                for item in items:
                    if item['language'] == 'pt_br':
                        text = item['text']
                        for entity in item['entities']:
                            entity_text = text[entity['start']:entity['end']]
                            entity_type = entity['entity']
                            text = text.replace(entity_text, f'[{entity_text}]({entity_type})')
                        if text not in train_items:
                            print('            intent_test: ', intent, '   text_test: ', text)
                            nlu_file.write(f'- {text}\n')
                        else:
                            overlap_count += 1
            print('train_count: ', train_count)
            print('test_count: ', test_count)
            print('overlap_count', overlap_count)


# BothubRepository(repo_name='odotonlogical_plan',
#                  repo_uuid='513f82e0-1768-4e65-a721-c35ae9bfd162',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='sac_viario',
#                  repo_uuid='792b3931-3da3-4f1c-94d0-db0a4028f4e4',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='susana',
#                  repo_uuid='1f86ecdf-4659-4a98-84bf-78d0ef9d3512',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='cadastracao_v4',
#                  repo_uuid='4bc1d618-759a-4814-bc8a-2c9faaff7d83',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='GCSuporte2',
#                  repo_uuid='b8855b7a-7e1e-4016-8084-1803eff787cf',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='nina_unfpa',
#                  repo_uuid='9a1b6604-50e6-4322-a0a2-1def0ee5d549',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='sei',
#                  repo_uuid='2a4d6a3a-a2ed-433e-a16c-f8b1e1749f10',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='sac_viario_sup_1',
#                  repo_uuid='e58c4aa5-c9d7-4b53-af07-4891abb08012',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='nina',
#                  repo_uuid='2b2e5826-b5a5-49bd-866e-3eb536456726',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

# BothubRepository(repo_name='susana',
#                  repo_uuid='1f86ecdf-4659-4a98-84bf-78d0ef9d3512',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()
#
# BothubRepository(repo_name='publicorganization',
#                  repo_uuid='cc8c6f95-7447-4d9b-a262-18bd258cae72',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()
#
# BothubRepository(repo_name='public_organization',
#                  repo_uuid='206a1c69-86e3-4f9e-a19d-15337ebeec2b',
#                  language='pt_br',
#                  auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

BothubRepository(repo_name='seec',
                 repo_uuid='38ba6a21-8235-439b-b807-b257d404ec64',
                 language='pt_br',
                 auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

BothubRepository(repo_name='HPB',
                 repo_uuid='9fd4033d-88f8-462b-9754-c3ae94b97939',
                 language='pt_br',
                 auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

BothubRepository(repo_name='enterprise',
                 repo_uuid='28df8c8e-5787-43e8-9da6-8bee3ad54582',
                 language='pt_br',
                 auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

BothubRepository(repo_name='comercial',
                 repo_uuid='1741cc9f-058f-438b-8d57-c312f4a9ee47',
                 language='pt_br',
                 auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

BothubRepository(repo_name='susaninha',
                 repo_uuid='e87dbc88-7454-4f84-9914-cf03d6be8116',
                 language='pt_br',
                 auth_token='Token 62d6ca792529e2bd9b97ba9425962c90f675579f').get_merged_train_test()

