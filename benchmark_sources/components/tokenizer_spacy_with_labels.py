from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer as RasaSpacyTokenizer


class SpacyTokenizer(RasaSpacyTokenizer):
    name = "tokenizer_spacy_with_labels"

    def train(self, training_data, config, **kwargs):
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.get("spacy_doc")))