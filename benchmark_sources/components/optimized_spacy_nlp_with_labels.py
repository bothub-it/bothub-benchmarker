import spacy
from rasa.nlu.utils.spacy_utils import SpacyNLP as RasaNLUSpacyNLP
from rasa.nlu.config import override_defaults


class SpacyNLP(RasaNLUSpacyNLP):
    name = "optimized_spacy_nlp_with_labels"

    @classmethod
    def load(
        cls, meta, model_dir=None, model_metadata=None, cached_component=None, **kwargs
    ):
        if cached_component:
            return cached_component
        nlp_language = spacy.load('pt', parser=False)
        cls.ensure_proper_language_model(nlp_language)
        return cls(meta, nlp_language)

    @classmethod
    def create(cls, component_config, config):
        component_config = override_defaults(cls.defaults, component_config)

        spacy_model_name = component_config.get("model")

        # if no model is specified, we fall back to the language string
        if not spacy_model_name:
            component_config["model"] = config.language
        nlp_language = spacy.load('pt_core_news_sm', parser=False)
        cls.ensure_proper_language_model(nlp_language)
        return cls(component_config, nlp_language)

    def train(self, training_data, config, **kwargs):
        for example in training_data.training_examples:
            example.set("spacy_doc", self.doc_for_text(example.text))
