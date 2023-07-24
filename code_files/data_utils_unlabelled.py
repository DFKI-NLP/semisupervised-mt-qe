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


class loadUnlabelledDatafromFile:
    
    ''' Loads data from file and returns dataframe'''
    
    def __init__(self,filepath_src= config.filePath_src,filePath_tar=config.filePath_tar):
        
        self.filePath_src = filepath_src
        self.filePath_tar=filePath_tar

        
        
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
    
    def createDfUnlabelled(self):
        

        column_names = ["source","target"]

        df = pd.DataFrame(columns=column_names,dtype=object)
        
        data_src = [i.strip() for i in self.load_data(self.filePath_src)]
        data_tar = [i.strip() for i in self.load_data(self.filePath_tar)]
            
        df = df.assign(source=data_src)
        df = df.assign(target = data_tar)

        return df
        


# In[5]:



class createTokenizedDfUnlabelled :
    '''Used for converting the input dataframe to tokenized dataframe'''
    
    def __init__(self,df,model_type):
        self.df = df
        self.model_type = model_type
    
    def convertDf(self):
        
        source_sentences = self.df["source"].tolist()
        target_sentences = self.df["target"].tolist()
        sentence_id = 0
        data = []
        
        
        for source_sentence, target_sentence in zip(source_sentences,target_sentences):
            
            for word in (source_sentence.split()):

                # Tokenize the word and count # of subwords the word is broken into

                
                if self.model_type == 'xlm':
                   
                    tokenized_word = config.TOKENIZER.tokenize(word)
                
                elif self.model_type == 'bert':
                    
                    tokenized_word = config.TOKENIZER_BERT.tokenize(word)
                
                n_subwords = len(tokenized_word)


                for i in range(n_subwords):
                    data.append([sentence_id, tokenized_word[i]])


            if self.model_type == 'xlm':
                data.append([sentence_id, "</s>"])
                data.append([sentence_id, "</s>"])
            
            elif self.model_type == 'bert':
                data.append([sentence_id, "[SEP]"])
            
            
            target_words = target_sentence.split()
            
            data.append([sentence_id, "का"]) #random gap token

            for word in target_words :

                if self.model_type == 'xlm':
                    tokenized_word = config.TOKENIZER.tokenize(word)
                
                elif self.model_type == 'bert':
                    tokenized_word = config.TOKENIZER_BERT.tokenize(word)
                
                
                n_subwords = len(tokenized_word)

                for i in range(n_subwords):
                    data.append([sentence_id, tokenized_word[i]])

                data.append([sentence_id, "का"]) # random gap token from vocab
            
            if self.model_type == 'xlm':
                data.append([sentence_id,'</s>'])
            elif self.model_type == 'bert':
                data.append([sentence_id,'[SEP]'])
            
            sentence_id += 1

        new_df=pd.DataFrame(data, columns=['sentence_id', 'words'])
        
        return new_df


# In[6]:

# class CompDatasetUnlabelled(Dataset):

#     def __init__(self, df,model_type):
        
#         self.df_data = df
#         self.model_type = model_type

#     def __getitem__(self, index):
        
        
        
#         temp_df = self.df_data.loc[self.df_data['sentence_id']==index]
        
#         tokens = temp_df['words'].tolist()
        
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
        
        
        
#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         attention_mask = torch.tensor(attention_mask, dtype=torch.long)
#         labels = [-100] * config.MAX_LEN
        
#         labels = torch.tensor(labels,dtype=torch.long)
#         sample = (input_ids, attention_mask,labels)
        
#         assert len(input_ids) == config.MAX_LEN
#         assert len(attention_mask) == config.MAX_LEN
#         assert len(labels) == config.MAX_LEN

#         return sample
    
#     def __len__(self):
#         return len(self.df_data.groupby(['sentence_id']))


class CompDatasetUnlabelled(Dataset):

    def __init__(self, df,model_type):
        
        self.df_data = df
        self.model_type = model_type

    def __getitem__(self, index):
        
        
        
        temp_df = self.df_data.loc[self.df_data['sentence_id']==index]
        
        tokens = temp_df['words'].tolist()
        
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
        
        labels = [-100] * config.MAX_LEN
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        labels = torch.tensor(labels,dtype=torch.long)
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




