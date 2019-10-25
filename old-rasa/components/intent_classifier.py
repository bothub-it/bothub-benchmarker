from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import typing
from builtins import zip
import os
import io
from future.utils import PY3
from typing import Any, Optional
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

import numpy as np

from rasa_nlu import utils
from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn

SKLEARN_MODEL_FILE_NAME = "intent_classifier_test.pkl"


def _sklearn_numpy_warning_fix():
    """Fixes unecessary warnings emitted by sklearns use of numpy.

    Sklearn will fix the warnings in their next release in ~ August 2018.

    based on https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array"""
    import warnings

    warnings.filterwarnings(module='sklearn*', action='ignore',
                            category=DeprecationWarning)


class intent_classifier_test(SklearnIntentClassifier):
    """Intent classifier using the sklearn framework"""

    name = "intent_classifier_sklearn2"

    def _create_classifier(self, num_threads, y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC

        C = self.component_config["C"]
        kernels = self.component_config["kernels"]
        gamma = self.component_config["gamma"]
        # dirty str fix because sklearn is expecting
        # str not instance of basestr...
        tuned_parameters = [{"C": C,
                             "gamma": gamma,
                             "kernel": [str(k) for k in kernels]}]

        # aim for 5 examples in each fold

        cv_splits = self._num_cv_splits(y)
        return GridSearchCV(SVC(C=1,
                                probability=True,
                                class_weight='balanced',
                                random_state=0),
                            param_grid=tuned_parameters,
                            n_jobs=num_threads,
                            cv=cv_splits,
                            scoring=self.component_config['scoring_function'],
                            verbose=1)
