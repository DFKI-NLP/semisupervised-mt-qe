#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install seqeval')
get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')


# In[ ]:





# In[3]:


import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import config


# In[ ]:


def ClassificationCost(output,target,mask):
    
    
    active_loss = mask.view(-1) == 1 #loss calculation for non padded tokens only (mask =1)
    active_logits = output.view(-1,2)
    
    active_labels = torch.where( # just append -100 for the padded tokens so its ignored when computing loss , no need now
        active_loss,             # since its handled in preprocessing only
        target.view(-1),
        torch.tensor(-100).type_as(target)    
    )
    try:
        class_0_weights =1/len(torch.where(active_labels==0)[0]) # trying to weight the labels as its unbalanced mostly
    
    except ZeroDivisionError:
        class_0_weights = float(1)
    

    try:
        class_1_weights =1/len(torch.where(active_labels==1)[0])
    
    except ZeroDivisionError:
        class_1_weights = float(1)
    
    weights_tensor = torch.tensor([class_0_weights,class_1_weights]).cuda()
    lfn = nn.CrossEntropyLoss(weight = weights_tensor)
    loss = lfn(active_logits,active_labels)
    
    return loss


# In[ ]:


def ConsistencyCost(student_output, teacher_output,mask,scale=10):
    
    assert len(student_output) == len(teacher_output)
    
    loss = nn.MSELoss()
    active_outputs = mask.view(-1) == 1 #loss calculation for non padded tokens only (mask =1)
    flattened_outputs_student = student_output.view(-1,2)
    flatenned_outputs_teacher = teacher_output.view(-1,2)

    active_outputs_teacher=flatenned_outputs_teacher[torch.where(mask.view(-1) == 1)]
    active_outputs_student=flattened_outputs_student[torch.where(mask.view(-1) == 1)]
    
    return scale * loss(active_outputs_student.view(-1),active_outputs_teacher.view(-1))
    


# In[ ]:


def update_teacher_params(student_model, teacher_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    
    teacher_params = teacher_model.parameters()
    student_params = student_model.parameters()
    
    assert sum(p.numel() for p in student_model.parameters()) == sum(p.numel() for p in teacher_model.parameters())
    
#     alpha = min(1 - 1 / (global_step + 1), alpha)
    
#     if global_step == 10 or global_step==20 or global_step == 30:
#         print(alpha)
    
    for teacher_param, student_param in zip(teacher_params, student_params):
            teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)

