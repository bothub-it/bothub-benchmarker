import spacy
from rasa_nlu.utils.spacy_utils import SpacyNLP as RasaNLUSpacyNLP


class SpacyNLP(RasaNLUSpacyNLP):
    name = "optimized_spacy_nlp_with_labels"

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        if cached_component:
            return cached_component
        nlp_language = spacy.load('pt_br', parser=False)
        component_meta = model_metadata.for_component(cls.name)
        cls.ensure_proper_language_model(nlp_language)
        return cls(component_meta, nlp_language)

    @classmethod
    def create(cls, cfg):
        component_conf = cfg.for_component(cls.name, cls.defaults)
        spacy_model_name = cfg.language
        component_conf["model"] = spacy_model_name
        nlp_language = spacy.load('pt_br', parser=False)
        cls.ensure_proper_language_model(nlp_language)
        return cls(component_conf, nlp_language)

    def train(self, training_data, config, **kwargs):
        for example in training_data.training_examples:
            example.set("spacy_doc", self.doc_for_text(example.text))
