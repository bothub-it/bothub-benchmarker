import logging
import re
from typing import Dict, List, Text, Union

from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.constants import DOCS_URL_TRAINING_DATA_NLU
import rasa.utils.io
import rasa.utils.io
import rasa.utils.common as common_utils

from ..nlp.preprocessing_base import PreprocessingBase

logger = logging.getLogger(__name__)


class RegexFeaturizerCustom(RegexFeaturizer):

    def _generate_lookup_regex(
        self, lookup_table: Dict[Text, Union[Text, List[Text]]]
    ) -> Text:
        """creates a regex out of the contents of a lookup table file"""
        lookup_elements = lookup_table["elements"]
        elements_to_regex = []

        # if it's a list, it should be the elements directly
        if isinstance(lookup_elements, list):
            elements_to_regex = lookup_elements
            common_utils.raise_warning(
                "Directly including lookup tables as a list is deprecated since Rasa "
                "1.6.",
                FutureWarning,
                docs=DOCS_URL_TRAINING_DATA_NLU + "#lookup-tables",
            )

        # otherwise it's a file path.
        else:

            try:
                f = open(lookup_elements, "r", encoding=rasa.utils.io.DEFAULT_ENCODING)
            except OSError:
                raise ValueError(
                    f"Could not load lookup table {lookup_elements}. "
                    f"Please make sure you've provided the correct path."
                )

            with f:

                preprocessor = PreprocessingBase()
                for line in f:
                    new_element = line.strip()
                    if new_element:
                        new_element = preprocessor.preprocess(new_element)
                        elements_to_regex.append(new_element)

        print("Loaded from lookup table: ", elements_to_regex)
        # sanitize the regex, escape special characters
        elements_sanitized = [re.escape(e) for e in elements_to_regex]

        # regex matching elements with word boundaries on either side
        regex_string = "(?i)(\\b" + "\\b|\\b".join(elements_sanitized) + "\\b)"
        return regex_string


