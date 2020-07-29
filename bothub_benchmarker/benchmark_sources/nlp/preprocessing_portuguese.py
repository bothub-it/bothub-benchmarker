from .preprocessing_interface import PreprocessingInterface
from .preprocessing_base import PreprocessingBase

import re


class PreprocessingPortuguese(PreprocessingInterface):

    def preprocess(self, phrase: str = None):
        
        if phrase == None:
            return

        phrase = PreprocessingBase().preprocess(phrase)

        phrase = phrase.replace("blza", "beleza")
        phrase = phrase.replace("flw", "falou")
        phrase = phrase.replace("vlw", "valeu")
        phrase = phrase.replace("ok", "okay")
        phrase = phrase.replace("tranks", "tranquilo")
        phrase = phrase.replace("mkt", "marketing")

        # set regex to "blz"
        blz_regex = r"\b(bl)z*\b"
        # set regex for "n":
        n_regex = r"\b(n|N)\1*\b"
        # set regex for "s":
        s_regex = r"\b(s|S)\1*\b"
        # set regex for "to":
        to_regex = r"\b(t)o*\b"
        # set regex for "ta":
        ta_regex = r"\b(t)o*\b"
        # set regex for "mkt":
        mkt_regex = r"\b(mkt)\b"
        # set regex for "ok":
        ok_regex = r"\b(ok)\b"
        # set regex for "bd":
        bd_regex = r"\b(bd)\b"

        # set replace words
        S_WORD = "sim"
        N_WORD = "nao"
        BLZ_WORD = "beleza"
        TO_WORD = "estou"
        TA_WORD = "esta"
        MKT_WORD = "marketing"
        OK_WORD = "okay"
        BD_WORD = "bom dia"

        # replace regex by S_WORD"
        phrase = re.sub(s_regex, S_WORD, phrase)
        # replace regex by N_WORD
        phrase = re.sub(n_regex, N_WORD, phrase)
        # replace regex by BLZ_WORD
        phrase = re.sub(blz_regex, BLZ_WORD, phrase)
        # replace regex by TO_WORD
        phrase = re.sub(to_regex, TO_WORD, phrase)
        # replace regex by TA_WORD
        phrase = re.sub(ta_regex, TA_WORD, phrase)
        # replace regex by MKT_WORD
        phrase = re.sub(mkt_regex, MKT_WORD, phrase)
        # replace regex by OK_WORD
        phrase = re.sub(ok_regex, OK_WORD, phrase)
        # replace regex by BD_WORD
        phrase = re.sub(bd_regex, BD_WORD, phrase)

        return phrase
