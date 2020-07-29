from .preprocessing_interface import PreprocessingInterface
from unidecode import unidecode


class PreprocessingBase(PreprocessingInterface):

    def preprocess(self, phrase: str = None):
        # removing accent and lowercasing characters
        return unidecode(phrase.lower())
