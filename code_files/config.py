#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')


# In[2]:

# 4800 + 8000 unlabelled =13000 - -100
import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 25
BASE_MODEL = "xlm-roberta-base" # multilingual bert can be tried as well
BASE_MODEL_LARGE = 'xlm-roberta-large'
BASE_MODEL_BERT = "distilbert-base-uncased"
random_seed = 42 # try more seeds - hyperparameter
test_split_ratio = 0.35
limit_eval_data = 2500
fix_train_size = 10000
gaussian_noise_std_student = 0.3 
gaussian_noise_std_teacher = 0.5 
alpha = 0.99
ratio = 0.5
threshold = 0.8
# consistency_cost_scale = 100
consistency_weight =1.0 # max weights till you want a ramp-up, will be this value after rampup epoch is reached
consistency_rampup = 500 # till what epochs rampup is needed
lr=2e-5 # hyperparameter
# MODEL_PATH = "model.bin"
# filePath_src = 's3://i540305-thesisdata/en-de-21/train.src'
# filePath_tar = 's3://i540305-thesisdata/en-de-21/train.mt'
# filePath_srcTags = 's3://i540305-thesisdata/en-de-21/train.source_tags'
# filePath_tarTags = 's3://i540305-thesisdata/en-de-21/train.tags'
# filePath_src_eval = 's3://i540305-thesisdata/en-de-21/dev.src'
# filePath_tar_eval = 's3://i540305-thesisdata/en-de-21/dev.mt'
# filePath_srcTags_eval='s3://i540305-thesisdata/en-de-21/dev.source_tags'
# filePath_tarTags_eval='s3://i540305-thesisdata/en-de-21/dev.tags'
filePath_src = '../data/train.src'
filePath_tar = '../data/train.mt'
filePath_srcTags = '../data/train.source_tags'
filePath_tarTags = '../data/train.tags'
filePath_src_eval = '../data/dev.src'
filePath_tar_eval = '../data/dev.mt'
filePath_srcTags_eval='../data/dev.source_tags'
filePath_tarTags_eval='../data/dev.tags'
filePath_src_backtranslated = '../data/backtranslated_fren.txt'
filePath_tar_backtranslated = '../data/backtranslated_fr_de.txt'

mname_en_fr = 'Helsinki-NLP/opus-mt-en-fr'
mname_fr_en = 'Helsinki-NLP/opus-mt-fr-en'
mname_de_fr = 'Helsinki-NLP/opus-mt-de-fr'
mname_fr_de = 'Helsinki-NLP/opus-mt-fr-de'


TOKENIZER = transformers.XLMRobertaTokenizer.from_pretrained(
    BASE_MODEL
#     do_lower_case=True
)
TOKENIZER_BERT = transformers.DistilBertTokenizer.from_pretrained(
    BASE_MODEL_BERT
#     do_lower_case=True
)


# In[ ]:




