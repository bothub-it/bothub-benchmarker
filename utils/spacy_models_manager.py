import os
import json


def installer():
    models = '../spacy_models'
    for model in os.listdir(models):
        if model.endswith('.zip'):
            pip_install_cmd = 'pip install ' + models + '/' + model
            spacy_model = model.split('-')[0]
            spacy_model = spacy_model.replace('-', '_')
            link_cmd = 'python -m spacy link ' + spacy_model + ' ' + spacy_model
            os.system(pip_install_cmd)
            os.system(link_cmd)


def generate_pipelines():
    models = '../spacy_models'
    out_directory = 'pipelines'
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    for model in os.listdir(models):
        if model.endswith('.zip'):
            spacy_model = model.split('-')[0]
            spacy_model = spacy_model.replace('-', '_')

            sklearn_name = spacy_model + '_sklearn.yml'
            sklearn_pipeline = 'language: "' + spacy_model + '"\n\n'
            sklearn_pipeline += 'pipeline:\n'
            sklearn_pipeline += '  - name: "benchmark_sources.components.optimized_spacy_nlp_with_labels.SpacyNLP"\n'
            sklearn_pipeline += '  - name: "benchmark_sources.components.tokenizer_spacy_with_labels.SpacyTokenizer"\n'
            sklearn_pipeline += '  - name: "RegexFeaturizer"\n'
            sklearn_pipeline += '  - name: "SpacyFeaturizer"\n'
            sklearn_pipeline += '  - name: "CRFEntityExtractor"\n'
            sklearn_pipeline += '  - name: "SpacyEntityExtractor"\n'
            sklearn_pipeline += '  - name: "benchmark_sources.components.crf_label_as_entity_extractor.CRFLabelAsEntityExtractor"\n'
            sklearn_pipeline += '  - name: "intent_classifier_sklearn"\n'
            with open(out_directory + '/' + sklearn_name, "w") as text_file:
                text_file.write(sklearn_pipeline)

            tensorflow_spacy = spacy_model + '_tensorflow_spacy.yml'
            tensorflow_spacy_pipeline = 'language: "' + spacy_model + '"\n\n'
            tensorflow_spacy_pipeline += 'pipeline:\n'
            tensorflow_spacy_pipeline += '  - name: "benchmark_sources.components.optimized_spacy_nlp_with_labels.SpacyNLP"\n'
            tensorflow_spacy_pipeline += '  - name: "benchmark_sources.components.tokenizer_spacy_with_labels.SpacyTokenizer"\n'
            tensorflow_spacy_pipeline += '  - name: "SpacyFeaturizer"\n'
            tensorflow_spacy_pipeline += '  - name: "EmbeddingIntentClassifier"\n'
            tensorflow_spacy_pipeline += '    similarity_type: "inner"\n'
            tensorflow_spacy_pipeline += '  - name: "CRFEntityExtractor"\n'
            tensorflow_spacy_pipeline += '  - name: "SpacyEntityExtractor"\n'
            tensorflow_spacy_pipeline += '  - name: "benchmark_sources.components.crf_label_as_entity_extractor.CRFLabelAsEntityExtractor"\n'
            with open(out_directory + '/' + tensorflow_spacy, "w") as text_file:
                text_file.write(tensorflow_spacy_pipeline)


def datasets_ranker(output_folder):
    # Get Datasets
    datasets = []
    for config in os.listdir(output_folder):
        datasets_folder = output_folder + '/' + config + '/Datasets_Results/'
        if os.path.exists(datasets_folder):
            for dataset in os.listdir(datasets_folder):
                datasets.append(dataset)

    for dataset in datasets:
        dataset_result = []
        for config in os.listdir(output_folder):
            datasets_folder = output_folder + '/' + config + '/Datasets_Results/'
            if os.path.exists(datasets_folder + dataset):
                with open(datasets_folder + dataset) as json_data:
                    d = json.load(json_data)
                    d['model'] = config
                    try:
                        del d['intent_evaluation']['report']
                    except KeyError:
                        pass
                    dataset_result.append(d)

        results_sorted = sorted(dataset_result, key=lambda k: k['intent_evaluation']['f1_score'], reverse=True)
        print(dataset)
        datasets_out = '/Overhaul_Results/Datasets_Results/'
        if not os.path.exists(output_folder + datasets_out):
            os.mkdir(output_folder + datasets_out)
        with open(output_folder + datasets_out + dataset + '_Ranked.json', 'w') as file:
            file.write(json.dumps(results_sorted, indent=4))


def benchmark_ranker():
    files_to_eval = ['Datasets_Mean_Result', 'Small_Datasets_Mean_Result', 'Medium_Datasets_Mean_Result', 'Big_Datasets_Mean_Result']
    output_folder = './benchmark_output/benchmark_all_data_v2'
    for file_to_eval in files_to_eval:
        # Load outputs
        results = []
        for output in os.listdir(output_folder):
            if os.path.exists(output_folder + '/' + output + '/' + file_to_eval):
                with open(output_folder + '/' + output + '/' + file_to_eval) as json_data:
                    d = json.load(json_data)
                    d['model'] = output
                    results.append(d)

        # Sort outputs
        results_sorted = sorted(results, key=lambda k: k['intent_evaluation']['f1_score'], reverse=True)

        # Save to file
        if not os.path.exists(output_folder + '/Overhaul_Results/'):
            os.mkdir(output_folder + '/Overhaul_Results/')
        with open(output_folder + '/Overhaul_Results/' + file_to_eval + '_Ranked.json', 'w') as file:
            file.write(json.dumps(results_sorted, indent=4))
    datasets_ranker(output_folder)


benchmark_ranker()
