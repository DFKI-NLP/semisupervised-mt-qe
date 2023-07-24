#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install seqeval')
get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')


# In[3]:


import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import config
from transformers import XLMRobertaForTokenClassification, XLMRobertaConfig ,BertModel, XLMRobertaTokenizer, XLMRobertaModel, BertForTokenClassification


# In[ ]:


class GaussianNoise(nn.Module):
    
    def __init__(self, stddev):
        
        super().__init__()
        
        self.stddev = stddev

    def forward(self, din):
        
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        
        return din


# In[4]:


class EntityModel(nn.Module):
    
    def __init__(self,std_gaussian=0.1,with_noise_layer = True, dropout_layer=False, dropout_prob = 0.3):
        
        super(EntityModel, self).__init__()
        
        self.std_gaussian = std_gaussian
        self.with_noise_layer = with_noise_layer
        self.bert = XLMRobertaModel.from_pretrained(config.BASE_MODEL,output_attentions = False, output_hidden_states = False)
        self.dropout_layer = dropout_layer
        self.dropout_prob = dropout_prob
        if self.with_noise_layer:
            self.noise = GaussianNoise(stddev=self.std_gaussian)
        if self.dropout_layer:
            self.bert_drop_1 = nn.Dropout(self.dropout_prob) # remove this or noise 
        self.out_tag = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, ids, attention_mask):
        
        outputs = self.bert(ids,
                            attention_mask = attention_mask,
                            return_dict=False)
        
        if (self.with_noise_layer):
            
            noise = self.noise(outputs[0]) # 256*768
            
            if self.dropout_layer:
                bo_tag = self.bert_drop_1(noise)
                tag = self.out_tag(bo_tag)
            else:
                tag = self.out_tag(noise)
        
        else:
            
            if self.dropout_layer:
                bo_tag = self.bert_drop_1(outputs[0])
                tag = self.out_tag(bo_tag)
            
            else :
                tag = self.out_tag(outputs[0])
#         tag = self.out_tag(bo_tag) # 256 * 2 
       
        softmax_prob = self.softmax(tag)
        
#         loss_tag = loss_fn(tag,labels,attention_mask)
        
        return softmax_prob,tag
#         return outputs[0], outputs[1]


# In[ ]:




