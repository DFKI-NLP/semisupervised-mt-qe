#!/usr/bin/env python
# coding: utf-8

# In[6]:

# This file includes various data augmentation schemes that we performed on our fixed datasets. 
import pandas as pd
import numpy as np
import random
import joblib
import torch
from nltk.corpus import wordnet
import config
from data_utils import loadDatafromFile,createTokenizedDf,CompDataset,createkfoldData,createDataloaders
import torch.nn as nn


# In[3]:


get_ipython().system('pip install nltk')


# In[40]:


class DataAugmentation:
    def __init__(self,dataframe,swap_words,syn_words, del_words_prob, num_sentences):
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
    'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 
    'himself', 'she', 'her', 'hers', 'herself', 
    'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 
    'whom', 'this', 'that', 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 
    'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'don', 
    'should', 'now', '']
        self.num_sentences = num_sentences
        self.dataframe = dataframe
        self.swap_words = swap_words
        self.syn_words = syn_words
        self.del_words_prob = del_words_prob
        
    #DataAugmentation methods
    
    def swap_word(self,new_words,labels_src):
        '''Helper function for random swap.''' 

        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return (new_words,labels_src)
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        labels_src[random_idx_1], labels_src[random_idx_2] = labels_src[random_idx_2], labels_src[random_idx_1]
        return (new_words, labels_src)

    
    def random_swap(self): # n is number of times to swap randomly 2 words

        '''Takes in input the initial dataframe read from the files and returns 
        modefied/increased dataframe with swapped word fro each source sentence and its corresponding token wise labels '''
        
        source_sentences  = list(self.dataframe.source)
        target_sentences = list(self.dataframe.target)
        labels_src = list(self.dataframe.src_tokens)
        labels_tar = list(self.dataframe.tar_tokens)
        source_sentences_temp =[]
        labels_sec_temp=[]
        i=0
    
        for sentences, labels in zip(source_sentences,labels_src):
            
            sentences = sentences.split()
            labels = labels.split()
            
            for _ in range(self.num_sentences):
                
                sentences_tmp = sentences.copy()
                labels_tmp = labels.copy()
                for _ in range(self.swap_words):
                    sentences_tmp, labels_tmp = self.swap_word(sentences_tmp,labels_tmp)
                
                assert(len(sentences_tmp) == len(labels_tmp))
                
                sentences_str = ' '.join(sentences_tmp)
                labels_str = ' '.join(labels_tmp)
                target_sentences.append(target_sentences[i])
                labels_tar.append(labels_tar[i])
                source_sentences_temp.append(sentences_str)
                labels_sec_temp.append(labels_str)
    #         break
            i+=1

        source_sentences.extend(source_sentences_temp)
        labels_src.extend(labels_sec_temp)

        column_names = ["source","target","src_tokens","tar_tokens"]
        df = pd.DataFrame(columns=column_names,dtype=object)
        df = df.assign(source=source_sentences)
        df = df.assign(target = target_sentences)
        df = df.assign(src_tokens = labels_src)
        df = df.assign(tar_tokens = labels_tar)

        return df


    
    def random_deletion(self):
        
        '''Takes input dataframe created after reading data files and the probabaility for deletion of random tokens in the
        source sentences and returns increased dataframe with combined orignal sentences and noisy sentences '''
    
        source_sentences  = list(self.dataframe.source)
        target_sentences = list(self.dataframe.target)
        labels_src = list(self.dataframe.src_tokens)
        labels_tar = list(self.dataframe.tar_tokens)
        senetences_temp=[]
        labels_temp= []
        #randomly delete words with probability p
        i=0
        for sentences, labels in zip(source_sentences,labels_src):
            
            sentences = sentences.split()
            labels = labels.split()
            
            if len(sentences) == 1:
                    pass
            
            
            for _ in range(self.num_sentences):
                
                source_sentences_temp=[]
                labels_sec_temp=[]
                
                for word,label in zip(sentences,labels):
                    r = random.uniform(0, 1)
                    if r > self.del_words_prob:
                        source_sentences_temp.append(word)
                        labels_sec_temp.append(label)
                if len(source_sentences_temp) == 0: #if you end up deleting all words, just return a random word
                    rand_int = random.randint(0, len(source_sentences_temp)-1)
                    source_sentences_temp.append(sentences[rand_int])
                    labels_sec_temp.append(labels[rand_int])
    
                sentences_str = ' '.join(source_sentences_temp)
                labels_str = ' '.join(labels_sec_temp)
                senetences_temp.append(sentences_str)
                labels_temp.append(labels_str)
                target_sentences.append(target_sentences[i])
                labels_tar.append(labels_tar[i])
    #         break
            i+=1
        source_sentences.extend(senetences_temp)
        labels_src.extend(labels_temp)    
        column_names = ["source","target","src_tokens","tar_tokens"]
        df = pd.DataFrame(columns=column_names,dtype=object)
        df = df.assign(source=source_sentences)
        df = df.assign(target = target_sentences)
        df = df.assign(src_tokens = labels_src)
        df = df.assign(tar_tokens = labels_tar)  


        return df
    

    def get_synonyms(self,word):
        
        '''Helper function for synonym replacement '''
        
        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)
    
    
    def synonym_replacement(self):
#     new_words = words.copy()
        source_sentences  = list(self.dataframe.source)
        target_sentences = list(self.dataframe.target)
        labels_src = list(self.dataframe.src_tokens)
        labels_tar = list(self.dataframe.tar_tokens)
        senetences_temp_lst=[]
        labels_temp_lst= []
        k=0
        for sentences,labels in zip(source_sentences,labels_src):
    #         dict={}
            
            sentences = sentences.split()
            labels_src_tmp = labels.split()
            
            
            for _ in range(self.num_sentences):
    
                sentences_tmp = sentences.copy()
                labels_tmp = labels_src_tmp.copy()
                random_word_list = list(set([word for word in sentences if word.lower() not in self.stop_words]))
                random.shuffle(random_word_list)
                num_replaced = 0
                for random_word in random_word_list:
                    synonyms = self.get_synonyms(random_word)
                    if len(synonyms) >= 1:
                        synonym = random.choice(list(synonyms))
                        synonym_len = len(synonym.split())
                        index_replaced_word = [i for i,item in enumerate(sentences_tmp) if item == random_word] # 2,7
                        labels_replaced_word = [labels_tmp[i] for i in index_replaced_word] # 0,1
                        index_replaced_word= [index+i*(synonym_len-1) for i,index in enumerate(index_replaced_word)]
                        for index,label in zip(index_replaced_word,labels_replaced_word):
                            for i in range(synonym_len-1):
                                labels_tmp.insert(index+i,label)
                        sentences_tmp = [synonym if word == random_word else word for word in sentences_tmp]


                        num_replaced += 1

                        if num_replaced >= self.syn_words: #only replace up to n words
                            break
                sentence = ' '.join(sentences_tmp)
                labels_splt = ' '.join(labels_tmp)
                senetences_temp_lst.append(sentence)
                labels_temp_lst.append(labels_splt)
                target_sentences.append(target_sentences[k])
                labels_tar.append(labels_tar[k])
            k+=1     

        source_sentences.extend(senetences_temp_lst)
        labels_src.extend(labels_temp_lst)
        column_names = ["source","target","src_tokens","tar_tokens"]
        df = pd.DataFrame(columns=column_names,dtype=object)
        df = df.assign(source=source_sentences)
        df = df.assign(target = target_sentences)
        df = df.assign(src_tokens = labels_src)
        df = df.assign(tar_tokens = labels_tar)

        return df


# In[41]:


# dataObj = loadDatafromFile(config.filePath_src,config.filePath_tar, config.filePath_srcTags,config.filePath_tarTags)
# df= dataObj.createDf() 
# augObj = DataAugmentation()
# augObj.synonym_replacement(df,2)


# In[21]:


# augObj = DataAugmentation()
# swapped_df = augObj.random_swap(df,2)


# In[22]:




# In[23]:




# In[33]:




# def synonym_replacement(dataframe, n):
# #     new_words = words.copy()
#     source_sentences  = list(dataframe.source)
#     target_sentences = list(dataframe.target)
#     labels_src = list(dataframe.src_tokens)
#     labels_tar = list(dataframe.tar_tokens)
#     senetences_temp=[]
#     labels_temp= []
#     k=0
#     for sentences,labels in zip(source_sentences,labels_src):
#         dict={}
#         sentences = sentences.split() # ['tarun', 'are', 'bad', 'for','health']
#         labels = labels.split()#['OK','BAD','OK','OK','OK']
#         for i, words in enumerate(sentences) :
#             dict[words] = i                      # {tarun:0, is : 1 , bad: 3 , for : 4, health : 5}
#         random_word_list = list(set([word for word in sentences if word not in config.stop_words]))
#         random.shuffle(random_word_list) # ['bad', 'are'...]
#         num_replaced = 0
#         for random_word in random_word_list: # ['bad', 'are'...]
#             synonyms = get_synonyms(random_word)
#             if len(synonyms) >= 1:
#                 synonym = random.choice(list(synonyms)) # bad -- > not good
#                 sentences = [synonym if word == random_word else word for word in sentences] # ['tarun', 'are', 'not good', 'for','health']
#                 synonym_len = len(synonym.split()) # 2 
#                 index_random_word = dict[random_word] # 2
#                 flag=0
#                 for key,values in dict.items():
#                     if key == random_word: # {tarun:0, is : 1 , bad: 2 , for : 3, health : 4}
#                         flag+=1
#                     if flag ==1:
#                         dict[key] = values+synonym_len-1 # {tarun:0, is : 1 , bad: 3 , for : 4, health : 5}
#                 label_synonym = labels[index_random_word] # 2 --> OK
#                 labels_append_synonym = [label_synonym] * synonym_len # ['OK','OK']
#                 labels_1 = labels[:index_random_word] # ['OK','BAD']
#                 labels_2 = labels[index_random_word+1:]#['OK','OK']
#                 labels_1.extend(labels_append_synonym)# ['OK','BAD','OK','OK']
#                 labels_1.extend(labels_2)# ['OK','BAD','OK','OK','OK','OK']
#                 labels = labels_1
#                 num_replaced += 1
  
#             if num_replaced >= n: #only replace up to n words
#                 break
#         sentence = ' '.join(sentences)
#         labels_splt = ' '.join(labels)
#         senetences_temp.append(sentence)
#         labels_temp.append(labels_splt)
#         target_sentences.append(target_sentences[k])
#         labels_tar.append(labels_tar[k])
# #         break
#         k+=1 
         
#     source_sentences.extend(senetences_temp)
#     labels_src.extend(labels_temp)
#     column_names = ["source","target","src_tokens","tar_tokens"]
#     df = pd.DataFrame(columns=column_names,dtype=object)
#     df = df.assign(source=source_sentences)
#     df = df.assign(target = target_sentences)
#     df = df.assign(src_tokens = labels_src)
#     df = df.assign(tar_tokens = labels_tar)

#     return df


# def get_synonyms(word):
#     synonyms = set()
#     for syn in wordnet.synsets(word): 
#         for l in syn.lemmas(): 
#             synonym = l.name().replace("_", " ").replace("-", " ").lower()
#             synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
#             synonyms.add(synonym) 
#     if word in synonyms:
#         synonyms.remove(word)
#     return list(synonyms)


# In[ ]:




