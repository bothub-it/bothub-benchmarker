import re
from .preprocessing_interface import PreprocessingInterface
from .preprocessing_base import PreprocessingBase


class PreprocessingEnglish(PreprocessingInterface):

    def preprocess(self, phrase: str = None):

        if phrase == None:
            return
        
        phrase = PreprocessingBase().preprocess(phrase)
        
        # set regex for "mkt":
        mkt_regex = r"\b(mkt)\b"
        # set regex for "ok":
        ok_regex = r"\b(ok)\b"
        # set regex for "ty":
        ty_regex = r"\b(ty)\b"
        # set regex for "thx":
        thx_regex = r"\b(thx)\b"
        # set regex for "tks":
        tks_regex = r"\b(tks)\b"
        # set regex for " 'm / 'mmmm ":
        am_regex = r"('m)m*\b"
        # set regex for " 'm / 'mmmm ":
        are_regex = r"('re)e*\b"
        # set regex for " *n't ":
        not_regex = r"(n't)\b"

        # set replace words
        MKT_WORD = "marketing"
        OK_WORD = "okay"
        TY_WORD = "thank you"
        AM_WORD = " am"
        ARE_WORD = " are"
        NOT_WORD = " not"

        # replace regex by MKT_WORD
        phrase = re.sub(mkt_regex, MKT_WORD, phrase)
        # replace regex by OK_WORD
        phrase = re.sub(ok_regex, OK_WORD, phrase)
        # replace regex by TY_WORD
        phrase = re.sub(ty_regex, TY_WORD, phrase)
        # replace regex by THX_WORD
        phrase = re.sub(thx_regex, TY_WORD, phrase)
        # replace regex by TKS_WORD
        phrase = re.sub(tks_regex, TY_WORD, phrase)
        # replace regex by AM_WORD
        phrase = re.sub(am_regex, AM_WORD, phrase)
        # replace regex by ARE_WORD
        phrase = re.sub(are_regex, ARE_WORD, phrase)
        # replace regex by NOT_WORD
        phrase = re.sub(not_regex, NOT_WORD, phrase)

        return phrase
