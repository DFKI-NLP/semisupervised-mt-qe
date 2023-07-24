#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seqeval')
get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')


# In[3]:


import torch
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import config as config
import pandas as pd
import s3fs
from torch.utils.data import Dataset, DataLoader
import random
from transformers import AdamW
import gc
from torch.utils.data import Dataset, DataLoader


# In[4]:


class loadDatafromFile:
    
    ''' Loads data from file and returns dataframe'''
    
    def __init__(self,filepath_src= config.filePath_src,filePath_tar=config.filePath_tar, filePath_srcTags=config.filePath_srcTags,
                 filePath_tarTags=config.filePath_tarTags):
        
        self.filePath_src = filepath_src
        self.filePath_tar=filePath_tar
        self.filePath_srcTags=filePath_srcTags
        self.filePath_tarTags=filePath_tarTags
        
        
#     def load_data(self,file):
        
#         fs = s3fs.S3FileSystem()
#         data=[]
#         with fs.open(file, encoding = "utf-8") as f:
#             for line in f:
#                 l = str(line, encoding='utf-8')
#     #             print(l)
#                 data.append(l)

#         f.close()
        
#         return data
        
    def load_data(self,file):
        
       
        data=[]
        with open(file, encoding = "utf-8") as f:
            for line in f:
                l = str(line)
    #             print(l)
                data.append(l)

        f.close()
        
        return data
    
    def createDf(self):
        
        column_names = ["source","target","src_tokens","tar_tokens"]
        df = pd.DataFrame(columns=column_names,dtype=object)
        data_src = [i.strip() for i in self.load_data(self.filePath_src)]
        data_tar = [i.strip() for i in self.load_data(self.filePath_tar)]
        data_srcTags = [i.strip() for i in self.load_data(self.filePath_srcTags)]
        data_tarTags = [i.strip() for i in self.load_data(self.filePath_tarTags)]

        df = df.assign(source=data_src)
        df = df.assign(target = data_tar)
        df = df.assign(src_tokens = data_srcTags)
        df = df.assign(tar_tokens = data_tarTags)
        
        return df
        


# In[5]:



class createTokenizedDf :
    '''Used for converting the input dataframe to tokenized dataframe'''
    
    def __init__(self,df,model_type):
        self.df = df
        self.model_type = model_type
    
    def convertDf(self):
        
        source_sentences = self.df["source"].tolist()
        source_tags = self.df["src_tokens"].tolist()
        target_sentences = self.df["target"].tolist()
        target_tags = self.df["tar_tokens"].tolist()
        sentence_id = 0
        data = []
        
        
        for source_sentence, source_tag_line, target_sentence, target_tag_line in zip(source_sentences, source_tags,
                                                                                      target_sentences, target_tags):
            for word, tag in zip(source_sentence.split(), source_tag_line.split()):

                # Tokenize the word and count # of subwords the word is broken into

                
                if self.model_type == 'xlm':
                   
                    tokenized_word = config.TOKENIZER.tokenize(word)
                
                elif self.model_type == 'bert':
                    
                    tokenized_word = config.TOKENIZER_BERT.tokenize(word)
                
                n_subwords = len(tokenized_word)


                for i in range(n_subwords):
                    data.append([sentence_id, tokenized_word[i],tag])


            if self.model_type == 'xlm':
                data.append([sentence_id, "</s>", "-100"])
                data.append([sentence_id, "</s>", "-100"])
            
            elif self.model_type == 'bert':
                data.append([sentence_id, "[SEP]", "-100"])
            
            
            target_words = target_sentence.split()
            target_tags = target_tag_line.split()
            
            data.append([sentence_id, "का", target_tags.pop(0)]) #random gap token

            for word in target_words :

                if self.model_type == 'xlm':
                    tokenized_word = config.TOKENIZER.tokenize(word)
                
                elif self.model_type == 'bert':
                    tokenized_word = config.TOKENIZER_BERT.tokenize(word)
                
                
                n_subwords = len(tokenized_word)

                for i in range(n_subwords):
                    data.append([sentence_id, tokenized_word[i],target_tags[0]])

                target_tags.pop(0)
                data.append([sentence_id, "का", target_tags.pop(0)]) # random gap token from vocab
            
            if self.model_type == 'xlm':
                data.append([sentence_id,'</s>','-100'])
            elif self.model_type == 'bert':
                data.append([sentence_id,'[SEP]','-100'])
            
            sentence_id += 1

        new_df=pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])
        new_df['labels'] = new_df['labels'].replace(['OK','BAD','-100'],[1,0,-100]) # Replacing labels with int tokens
        
        return new_df


# In[6]:


# class CompDataset(Dataset):

#     def __init__(self, df,model_type):
        
#         self.df_data = df
#         self.model_type = model_type

#     def __getitem__(self, index):
        
        
        
#         temp_df = self.df_data.loc[self.df_data['sentence_id']==index]
        
#         tokens = temp_df['words'].tolist()
#         labels = temp_df['labels'].tolist()
        
#         input_ids =[]
#         attention_mask =[]
        
        
#         if self.model_type == 'xlm':
#             input_ids = [0] + config.TOKENIZER.convert_tokens_to_ids(tokens) # adding <s> token
#         elif self.model_type == 'bert':
#             input_ids = [101] + config.TOKENIZER_BERT.convert_tokens_to_ids(tokens)
        
#         input_len=len(input_ids)
        
#         if self.model_type == 'xlm':
#             input_ids.extend([1] *(config.MAX_LEN-input_len)) # padding tokens
#         elif self.model_type == 'bert':
#             input_ids.extend([0] *(config.MAX_LEN-input_len)) # padding tokens
            
#         attention_mask.extend([1] * input_len)
#         attention_mask.extend([0] * (config.MAX_LEN-input_len)) # padding tokens
        
#         labels = [-100] + labels
#         labels.extend([-100] * (config.MAX_LEN-input_len))
        
        
#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         attention_mask = torch.tensor(attention_mask, dtype=torch.long)
#         labels = torch.tensor(labels, dtype=torch.long)
# #         labels = torch.tensor(labels)
#         sample = (input_ids, attention_mask,labels)
        
#         assert len(input_ids) == config.MAX_LEN
#         assert len(attention_mask) == config.MAX_LEN
#         assert len(labels) == config.MAX_LEN


#         return sample
    
#     def __len__(self):
#         return len(self.df_data.groupby(['sentence_id']))

class CompDataset(Dataset):

    def __init__(self, df,model_type):
        
        self.df_data = df
        self.model_type = model_type

    def __getitem__(self, index):
        
        
        
        temp_df = self.df_data.loc[self.df_data['sentence_id']==index]
        
        tokens = temp_df['words'].tolist()
        labels = temp_df['labels'].tolist()
        
        input_ids =[]
        attention_mask =[]
        
        
        if self.model_type == 'xlm':
            input_ids = [0] + config.TOKENIZER.convert_tokens_to_ids(tokens) # adding <s> token
        
        elif self.model_type == 'bert':
            input_ids = [101] + config.TOKENIZER_BERT.convert_tokens_to_ids(tokens)
        
        input_len=len(input_ids)
        
        if self.model_type == 'xlm':
            
            if input_len >= config.MAX_LEN:
                input_ids = input_ids[:config.MAX_LEN]
            
            else:
                input_ids.extend([1] *(config.MAX_LEN-input_len)) # padding tokens
        
        elif self.model_type == 'bert':
            
            if input_len >= config.MAX_LEN:
                input_ids = input_ids[:config.MAX_LEN]
            
            else:
                input_ids.extend([0] *(config.MAX_LEN-input_len)) # padding tokens
        
        if input_len >= config.MAX_LEN:
            attention_mask.extend([1] * (config.MAX_LEN))
        
        else:
            attention_mask.extend([1] * input_len)
            attention_mask.extend([0] * (config.MAX_LEN-input_len)) # padding tokens
            
        labels = [-100] + labels
        
        if input_len >= config.MAX_LEN :
            labels = labels[:config.MAX_LEN]
        
        else:
            labels.extend([-100] * (config.MAX_LEN-input_len))
        
#         if len(input_ids) > config.MAX_LEN : # for the cases if max_len is small like 128 
            
#             input_ids = input_ids[:config.MAX_LEN]
#             attention_mask=attention_mask[:config.MAX_LEN]
#             labels= labels[:config.MAX_LEN]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
#         labels = torch.tensor(labels)
        sample = (input_ids, attention_mask,labels)
        
        assert len(input_ids) == config.MAX_LEN
        assert len(attention_mask) == config.MAX_LEN
        assert len(labels) == config.MAX_LEN


        return sample
    
    def __len__(self):
        return len(self.df_data.groupby(['sentence_id']))





# In[10]:


class createkfoldData():
    
    def __init__(self,dataframe):
        self.dataframe = dataframe
        
    def get_kfoldIndexes():

        kf = KFold(n_splits=config.FOLDS)
        train_df_list = []
        val_df_list = []
        fold_list = list(kf.split(self.dataframe))

        for i, fold in enumerate(fold_list):

        # map the train and val index values to dataframe rows
            df_train = self.dataframe[self.dataframe.index.isin(fold[0])]
            df_val = self.dataframe[self.dataframe.index.isin(fold[1])]
            df_train = df_train.reset_index(drop=True)
            df_val = df_val.reset_index(drop=True)
            train_df_list.append(df_train)
            val_df_list.append(df_val)

        return train_df_list,val_df_list
    #     print(len(train_list))
    #     print(len(val_list))
    


# In[11]:


class createDataloaders():
    
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
    def createDataloaders(self):
        data_loader = torch.utils.data.DataLoader(self.dataset,batch_size = self.batch_size,shuffle = True, num_workers = 1) # for data to be returned in batches for batch grad-descent
    # one fold is now divided into 4 batches that can be accessed with any iterator like for ,etc
        return data_loader


# In[ ]:




