import os
import json


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
    output_folder = './benchmark_output/benchmark_legacy'
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
