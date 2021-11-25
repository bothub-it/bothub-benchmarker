from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from typing import List, Type
from rasa.nlu.components import Component


class LanguageModelFeaturizerCustom(LanguageModelFeaturizer):
    """Featurizer using transformer based language models.

    Uses the output of HFTransformersNLP component to set the sequence and sentence
    level representations for dense featurizable attributes of each message object.
    """
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return []
