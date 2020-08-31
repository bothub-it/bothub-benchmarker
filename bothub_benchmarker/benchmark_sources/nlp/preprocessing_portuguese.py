from .preprocessing_interface import PreprocessingInterface
from .preprocessing_base import PreprocessingBase

import re


class PreprocessingPortuguese(PreprocessingInterface):

    def preprocess(self, phrase: str = None):

        if phrase is None:
            return

        phrase = PreprocessingBase().preprocess(phrase)

        contractions = {
            "oi": r"\b((o+)(i+)(e*))\b",
            "sim": r"\b(s|S)+\b",
            "nao": r"\b(n|N)+\b",
            "beleza": r"\b(blz)z*a*\b",
            "estou": r"\b(t)o+\b",
            "esta": r"\b(t)a+\b",
            "marketing": r"\b(mkt)\b",
            "okay": r"\b(ok(a|e)*(y*))\b",
            "bom dia": r"\b(bd)\b",
            "falou": r"\b(f(a*)l(o*)(w|u)+(s*))\b",
            "valeu": r"\b(v(a*)l(e*)(w|u)+(s*))\b",
            "tranquilo": r"\b(tranks)\b"
        }

        for word in contractions.keys():
            phrase = re.sub(contractions[word], word, phrase)

        return phrase
