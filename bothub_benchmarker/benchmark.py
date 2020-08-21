import json
import posixpath
import os
from timeit import default_timer as timer
from rasa.nlu.test import *
from rasa.nlu.components import Component
from bothub_benchmarker.utils import upload_folder_to_bucket
# from false_positive_benchmark import false_positive_benchmark


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


PRETRAINED_EXTRACTORS = {"DucklingHTTPExtractor", "SpacyEntityExtractor"}
logger = logging.getLogger(__name__)
IntentMetrics = Dict[Text, List[float]]
EntityMetrics = Dict[Text, Dict[Text, List[float]]]

IntentEvaluationResult = namedtuple(
    "IntentEvaluationResult", "intent_target intent_prediction message confidence"
)

CVEvaluationResult = namedtuple("Results", "train test")

EntityEvaluationResult = namedtuple(
    "EntityEvaluationResult", "entity_targets entity_predictions tokens message"
)


# Override
def targets_predictions_from(
    results: Union[
        List[IntentEvaluationResult], List[ResponseSelectionEvaluationResult]
    ],
    target_key: Text,
    prediction_key: Text,
) -> Iterator[Iterable[Optional[Text]]]:
    return zip(*[(getattr(r, target_key), getattr(r, prediction_key)) for r in results])


# Override
def evaluate_intents(
    intent_results: List[IntentEvaluationResult],
    output_directory: Optional[Text],
    successes: bool,
    errors: bool
) -> Dict:  # pragma: no cover

    # remove empty intent targets
    num_examples = len(intent_results)
    intent_results = remove_empty_intent_examples(intent_results)

    logger.info(
        "Intent Evaluation: Only considering those "
        "{} examples that have a defined intent out "
        "of {} examples".format(len(intent_results), num_examples)
    )

    target_intents, predicted_intents = targets_predictions_from(
        intent_results, "intent_target", "intent_prediction"
    )

    report, precision, f1, accuracy = get_evaluation_metrics(
        target_intents, predicted_intents, output_dict=True
    )
    if isinstance(report, str):
        log_evaluation_table(report, precision, f1, accuracy)

    if successes:
        successes_filename = "intent_successes.json"
        if output_directory:
            successes_filename = os.path.join(output_directory, successes_filename)
        # save classified samples to file for debugging
        # collect_nlu_successes(intent_results, successes_filename)

    if errors:
        errors_filename = "intent_errors.json"
        if output_directory:
            errors_filename = os.path.join(output_directory, errors_filename)
        # log and save misclassified samples to file for debugging
        # collect_nlu_errors(intent_results, errors_filename)

    predictions = [
        {
            "text": res.message,
            "intent": res.intent_target,
            "predicted": res.intent_prediction,
            "confidence": res.confidence,
        }
        for res in intent_results
    ]

    return {
        "predictions": predictions,
        "report": report,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def save_result_by_group(datasets_results, n_fold, out_config_directory, datasets_names):
    big_results = []
    big_names = []
    medium_results = []
    medium_names = []
    small_results = []
    small_names = []
    size = len(datasets_results)
    for i in range(size):
        size = int(datasets_results[i]['intent_evaluation']['report']['weighted avg']['support']) * int(n_fold)
        if size < 300:
            small_results.append(datasets_results[i])
            small_names.append(datasets_names[i])
        elif size < 700:
            medium_results.append(datasets_results[i])
            medium_names.append(datasets_names[i])
        else:
            big_results.append(datasets_results[i])
            big_names.append(datasets_names[i])
    if len(small_results) > 0:
        small_result = sum_results(small_results)
        small_result['datasets'] = small_names
        utils.write_json_to_file(out_config_directory + 'Small_Datasets_Mean_Result', small_result)
    if len(medium_results) > 0:
        medium_result = sum_results(medium_results)
        medium_result['datasets'] = medium_names
        utils.write_json_to_file(out_config_directory + 'Medium_Datasets_Mean_Result', medium_result)
    if len(big_results) > 0:
        big_result = sum_results(big_results)
        big_result['datasets'] = big_names
        utils.write_json_to_file(out_config_directory + 'Big_Datasets_Mean_Result', big_result)


def sum_results(results, collect_report=False, has_entity_eval=False):
    intent_eval = results[0]['intent_evaluation']

    general_result = {
        "intent_evaluation": {
            "precision": intent_eval['precision'],
            "f1_score": intent_eval['f1_score'],
            "accuracy": intent_eval['accuracy']
        },
        "datasets": []
    }
    if has_entity_eval:
        entity_algorithm = list(results[0]['entity_evaluation'].keys())[0]
        entity_eval = results[0]['entity_evaluation'][entity_algorithm]
        general_result['entity_evaluation'] = {
            entity_algorithm: {
                "precision": entity_eval['precision'],
                "f1_score": entity_eval['f1_score'],
                "accuracy": entity_eval['accuracy']
            }
        },

    if collect_report:
        general_result['intent_evaluation']['report'] = intent_eval['report']

    # Sum of elements
    size = len(results)
    intent_result_divider = 1
    for result in results[1:]:
        intent_eval = result['intent_evaluation']
        # precision, f1_score and accuracy from intent eval
        general_result['intent_evaluation']['precision'] += intent_eval['precision']
        general_result['intent_evaluation']['f1_score'] += intent_eval['f1_score']
        general_result['intent_evaluation']['accuracy'] += intent_eval['accuracy']

        # report from intent eval
        if collect_report:
            report = result['intent_evaluation']['report']
            for intent in report:
                try:
                    for field in report[intent]:
                        general_result['intent_evaluation']['report'][intent][field] += report[intent][field]
                    intent_result_divider += 1
                except Exception as e:
                    print('exception:')
                    print(str(e))
                    print(json.dumps(report, indent=2))
                    pass
            if has_entity_eval:
                entity_eval = result['entity_evaluation'][entity_algorithm]

        # precision, f1_score and accuracy from entity eval
        if has_entity_eval:
            general_result['entity_evaluation'][entity_algorithm]['precision'] += entity_eval['precision']
            general_result['entity_evaluation'][entity_algorithm]['f1_score'] += entity_eval['f1_score']
            general_result['entity_evaluation'][entity_algorithm]['accuracy'] += entity_eval['accuracy']

    # Mean of elements
    general_result['intent_evaluation']['precision'] /= size
    general_result['intent_evaluation']['f1_score'] /= size
    general_result['intent_evaluation']['accuracy'] /= size

    if collect_report:
        for intent in general_result['intent_evaluation']['report']:
            try:
                for field in general_result['intent_evaluation']['report'][intent]:
                    general_result['intent_evaluation']['report'][intent][field] /= size
            except TypeError:
                pass
    if has_entity_eval:
        general_result['entity_evaluation'][entity_algorithm]['precision'] /= size
        general_result['entity_evaluation'][entity_algorithm]['f1_score'] /= size
        general_result['entity_evaluation'][entity_algorithm]['accuracy'] /= size

    return general_result


def generate_folds(
    n: int, td: TrainingData
) -> Iterator[Tuple[TrainingData, TrainingData]]:
    """Generates n cross validation folds for training data td."""

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=3)
    x = td.intent_examples
    y = [example.get("intent") for example in x]
    for i_fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        logger.debug("Fold: {}".format(i_fold))
        train = [x[i] for i in train_index]
        test = [x[i] for i in test_index]
        yield (
            TrainingData(
                training_examples=train,
                entity_synonyms=td.entity_synonyms,
                regex_features=td.regex_features,
            ),
            TrainingData(
                training_examples=test,
                entity_synonyms=td.entity_synonyms,
                regex_features=td.regex_features,
            ),
        )


def run_benchmark(data_path, n_folds, trainer):  # pragma: no cover
    """Evaluate intent classification and entity extraction."""
    data = training_data.load_data(data_path)
    # data_to_evaluate = drop_intents_below_freq(data_to_evaluate, cutoff=5)
    # get the metadata config from the package data_to_evaluate
    count = 0
    results = []
    for train, test in generate_folds(n_folds, data):
        count += 1
        interpreter = trainer.train(train)

        interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

        result = {
            "intent_evaluation": None,
            "entity_evaluation": None,
            "response_selection_evaluation": None,
        }  # type: Dict[Text, Optional[Dict]]

        intent_results, response_selection_results, entity_results, = get_eval_data(
            interpreter, test
        )
        successes = True
        errors = True
        if intent_results:
            logger.info("Intent evaluation results:")
            result["intent_evaluation"] = evaluate_intents(
                intent_results, None, successes, errors
            )

        if response_selection_results:
            logger.info("Response selection evaluation results:")
            result["response_selection_evaluation"] = evaluate_response_selections(
                response_selection_results, report_folder=None
            )

        if entity_results:
            logger.info("Entity evaluation results:")
            extractors = get_entity_extractors(interpreter)
            result["entity_evaluation"] = evaluate_entities(
                entity_results, extractors, None, False, False
            )
        results.append(result)
    return results


def remove_pretrained_extractors(pipeline: List[Component]) -> List[Component]:
    """Removes pretrained extractors from the pipeline so that entities
       from pre-trained extractors are not predicted upon parsing"""
    pipeline = [c for c in pipeline if c.name not in PRETRAINED_EXTRACTORS]
    return pipeline


def benchmark(out_directory, config_directory, dataset_directory, n_folds=3, bucket=None, job_id=None):
    start = timer()

    out_directory_temp = out_directory
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    else:
        count = 0
        while os.path.exists(out_directory_temp):
            out_directory_temp = out_directory + str(count)
            count += 1
        os.mkdir(out_directory_temp)

    config_size = len(os.listdir(config_directory))
    count_config = 0
    for config_filename in os.listdir(config_directory):
        count_config += 1
        print('######################################')
        print('CURRENT CONFIG :', config_filename, ' PROGRESS:', count_config, '/', config_size)
        print('######################################')
        start_config = timer()
        if config_filename.endswith(".yml"):
            config_path = posixpath.join(config_directory, config_filename)
            config_name = config_filename.split('.')[0]
            out_config_directory = posixpath.join(out_directory_temp, config_name)
            if not os.path.exists(out_config_directory):
                os.mkdir(out_config_directory)
            datasets_dir_out = 'Datasets_Results/'
            if not os.path.exists(out_config_directory + datasets_dir_out):
                os.mkdir(out_config_directory + datasets_dir_out)

            nlu_config = config.load(config_path)
            try:
                trainer = Trainer(nlu_config)
                trainer.pipeline = remove_pretrained_extractors(trainer.pipeline)
            except OSError:
                raise
            datasets_results = []
            datasets_names = []

            for dataset_filename in os.listdir(dataset_directory):
                if dataset_filename.endswith(".json") or dataset_filename.endswith(".md"):
                    dataset_path = os.path.join(dataset_directory, dataset_filename)
                    dataset_name = dataset_filename.split('.')[0]

                    cross_val_results = run_benchmark(dataset_path, n_folds, trainer)
                    # utils.write_json_to_file('new_result_test', cross_val_results)

                    dataset_result = sum_results(cross_val_results, collect_report=True)
                    utils.write_json_to_file(out_config_directory + datasets_dir_out + dataset_name + '_Benchmark',
                                             dataset_result)
                    datasets_results.append(dataset_result)
                    datasets_names.append(dataset_filename)
            save_result_by_group(datasets_results, n_folds, out_config_directory, datasets_names)
            overhaul_result = sum_results(datasets_results)
            end_config = timer()
            overhaul_result['time'] = str(end_config)
            utils.write_json_to_file(out_config_directory + 'Datasets_Mean_Result', overhaul_result)
            if bucket is not None:
                upload_folder_to_bucket(bucket, out_directory, posixpath.join('results', out_directory))
    end = timer()
    logger.info("Finished evaluation in: " + str(end - start))


def set_tensorboard(nlu_config, out_directory, eval_examples=100):
    nlu_config = nlu_config.as_dict()

    for item in nlu_config['pipeline']:
        if 'DIETClassifier' in item['name']:
            item['tensorboard_log_directory'] = os.path.join(out_directory, 'Tensorboard')
            print("TENSORBOARD EVAL EXAMPLES SET TO: ", eval_examples)
            item['evaluate_on_number_of_examples'] = eval_examples
            item['evaluate_every_number_of_epochs'] = 5
            item['tensorboard_log_level'] = 'epoch'

            break

    return RasaNLUModelConfig(nlu_config)


def tensorboard_benchmark(out_directory, config_directory, dataset_directory):
    start = timer()

    out_directory_temp = out_directory
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    else:
        count = 0
        while os.path.exists(out_directory_temp):
            out_directory_temp = out_directory + str(count)
            count += 1
        os.mkdir(out_directory_temp)

    config_size = len(os.listdir(config_directory))
    count_config = 0
    for config_filename in os.listdir(config_directory):
        count_config += 1
        print('######################################')
        print('CURRENT CONFIG :', config_filename, ' PROGRESS:', count_config, '/', config_size)
        print('######################################')
        start_config = timer()
        if config_filename.endswith(".yml"):
            config_path = os.path.join(config_directory, config_filename)
            config_name = config_filename.split('.')[0]
            out_config_directory = out_directory_temp + '/' + config_name + '/'
            if not os.path.exists(out_config_directory):
                os.mkdir(out_config_directory)
            datasets_dir_out = 'Datasets_Results/'
            if not os.path.exists(out_config_directory + datasets_dir_out):
                os.mkdir(out_config_directory + datasets_dir_out)

            for dataset_filename in os.listdir(dataset_directory):

                if dataset_filename.endswith(".json") or dataset_filename.endswith(".md"):
                    dataset_path = os.path.join(dataset_directory, dataset_filename)
                    dataset_name = dataset_filename.split('.')[0]

                    print(dataset_path)
                    data = training_data.load_data(dataset_path)
                    eval_data_size = int(len(data.intent_examples)*0.25)

                    nlu_config = config.load(config_path)

                    nlu_config = set_tensorboard(nlu_config,
                                                 os.path.join(out_config_directory, datasets_dir_out, dataset_name),
                                                 eval_data_size)

                    try:
                        trainer = Trainer(nlu_config)
                        trainer.pipeline = remove_pretrained_extractors(trainer.pipeline)
                    except OSError:
                        raise

                    trainer.train(data)

            end_config = timer()

    end = timer()
    logger.info("Finished evaluation in: " + str(end - start))


if __name__ == '__main__':
    print("start benchmark")
    out_directory = 'benchmark_output_test1'
    config_directory = 'benchmark_sources/configs/'
    dataset_directory = 'benchmark_sources/data_to_evaluate/'
    tensorboard_benchmark(out_directory, config_directory, dataset_directory)
    # false_positive_dataset_directory = 'benchmark_sources/oldvsoldold'
    # false_positive_benchmark(out_directory, config_directory, false_positive_dataset_directory)
