#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from tqdm import tqdm
from seqeval.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef,f1_score,classification_report


# In[3]:


def train_fn(data_loader, model, optimizer, scheduler):
    model.train()
    total_train_loss = 0
    lst_active_labels = []
    lst_active_preds = []
    
    for batch in tqdm(data_loader, total = len(data_loader)):
        
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        outputs = model(b_input_ids, 
                        attention_mask=b_input_mask,
                        labels=b_labels)

#         print(outputs[1].grad_fn)
        loss = outputs[0]
#         print(loss.grad_fn)
#         loss.requires_grad =True 
        
        loss.backward()
        
        total_train_loss = total_train_loss + loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
       
        labels = b_labels.view(-1) 
        active_logits = outputs[1].view(-1, 2)
        flattened_predictions = torch.argmax(active_logits, axis=1)

        active_accuracy = labels.view(-1) != -100
        labels_tmp = torch.masked_select(labels, active_accuracy) 
        pred_tmp = torch.masked_select(flattened_predictions, active_accuracy) 
        lst_active_labels.extend(labels_tmp.tolist())
        lst_active_preds.extend(pred_tmp.tolist())
    
    avg_f1_score_0=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 0)
    avg_f1_score_1=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 1)     
    avg_accuracy_score=accuracy_score(lst_active_labels,lst_active_preds)
    avg_mcc_score=matthews_corrcoef(lst_active_labels,lst_active_preds)
   
    print('Average F1 Training score for class 0 :' ,avg_f1_score_0)
    print('Average F1 Training score for class 1 :' ,avg_f1_score_1)
    print('Average Accuracy Training score  :' ,avg_accuracy_score)
    print('Average mcc Training score  :' ,avg_mcc_score)
    
    
    return (float(total_train_loss / len(data_loader)),avg_f1_score_0,avg_f1_score_1, avg_accuracy_score, avg_mcc_score,lst_active_labels,lst_active_preds)


# In[4]:


def eval_fn(data_loader, model):
    model.eval()
    total_val_loss = 0
    lst_active_preds = []
    lst_active_labels = [] 

    for  batch in tqdm(data_loader,total=len(data_loader)):
        
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda()
#         torch.set_grad_enabled(False)
        outputs = model(b_input_ids, 
                        attention_mask=b_input_mask,
                        labels=b_labels)

            # Get the loss from the outputs tuple: (loss, logits)
        loss = outputs[0]
        # Convert the loss from a torch tensor to a number.
        # Calculate the total loss.
        total_val_loss = total_val_loss + loss.item()
        labels = b_labels.view(-1)
        active_logits = outputs[1].view(-1,2)
        flattened_preditions = torch.argmax(active_logits, axis=1)
        active_loss = b_labels.view(-1) != -100
        labels_tmp = torch.masked_select(labels,active_loss)
        pred_tmp = torch.masked_select(flattened_preditions,active_loss)
        lst_active_labels.extend(labels_tmp.tolist())
        lst_active_preds.extend(pred_tmp.tolist())
        
    avg_f1_score_0=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 0)
    avg_f1_score_1=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 1)
    avg_accuracy_score=accuracy_score(lst_active_labels,lst_active_preds)
    avg_mcc_score = matthews_corrcoef(lst_active_labels,lst_active_preds)
    
        
    print('Average F1 Validation score for class 0 :' ,avg_f1_score_0)
    print('Average F1 Validation score for class 1 :' ,avg_f1_score_1)
    print('Average Accuracy Validation score  :' ,avg_accuracy_score)
    print('Average mcc Validation score  :' ,avg_mcc_score)

        
    return (float(total_val_loss/len(data_loader)),avg_f1_score_0,avg_f1_score_1,avg_accuracy_score,
            avg_mcc_score,lst_active_labels,lst_active_preds)


# In[ ]:





# In[ ]:





# In[ ]:




