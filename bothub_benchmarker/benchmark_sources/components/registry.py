import logging

# Explicitly set logging level for this module before any import
# because otherwise it logs tensorflow/pytorch versions
logging.getLogger("transformers.file_utils").setLevel(logging.WARNING)

from transformers import (
    TFBertModel,
    TFOpenAIGPTModel,
    TFGPT2Model,
    TFXLNetModel,
    # TFXLMModel,
    TFDistilBertModel,
    TFRobertaModel,
    BertTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    XLNetTokenizer,
    # XLMTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
)
from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import (
    bert_tokens_pre_processor,
    gpt_tokens_pre_processor,
    xlnet_tokens_pre_processor,
    roberta_tokens_pre_processor,
    bert_embeddings_post_processor,
    gpt_embeddings_post_processor,
    xlnet_embeddings_post_processor,
    roberta_embeddings_post_processor,
    bert_tokens_cleaner,
    openaigpt_tokens_cleaner,
    gpt2_tokens_cleaner,
    xlnet_tokens_cleaner,
)

language_to_model = {
    "en": "bert_english",
    "pt_br": "bert_portuguese",
    "multilang": "bert_multilang",
    "es": "bert_spanish",
}

from_pt_dict = {
    "bert_portuguese": True,
    "bert_spanish": True,
}

model_class_dict = {
    "bert_english": TFBertModel,
    "bert_portuguese": TFBertModel,
    "bert_multilang": TFBertModel,
    "bert_spanish": TFBertModel,
}

model_tokenizer_dict = {
    "bert_english": BertTokenizer,
    "bert_portuguese": BertTokenizer,
    "bert_multilang": BertTokenizer,
    "bert_spanish": BertTokenizer,
}

model_weights_defaults = {
    "bert_english": "bert-base-uncased",
    "bert_portuguese": "neuralmind/bert-base-portuguese-cased",
    "bert_multilang": "bert-base-multilingual-uncased",
    "bert_spanish": "dccuchile/bert-base-spanish-wwm-uncased",
}

model_special_tokens_pre_processors = {
    "bert_english": bert_tokens_pre_processor,
    "bert_portuguese": bert_tokens_pre_processor,
    "bert_multilang": bert_tokens_pre_processor,
    "bert_spanish": bert_tokens_pre_processor,
}

model_tokens_cleaners = {
    "bert_english": bert_tokens_cleaner,
    "bert_portuguese": bert_tokens_cleaner,
    "bert_multilang": bert_tokens_cleaner,
    "bert_spanish": bert_tokens_cleaner,
}

model_embeddings_post_processors = {
    "bert_english": bert_embeddings_post_processor,
    "bert_portuguese": bert_embeddings_post_processor,
    "bert_multilang": bert_embeddings_post_processor,
    "bert_spanish": bert_embeddings_post_processor,
}

model_config_url = {
    "bert_portuguese": "https://s3.amazonaws.com/models.huggingface.co/bert/neuralmind/bert-base-portuguese-cased/config.json",
    "bert_english": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
}
