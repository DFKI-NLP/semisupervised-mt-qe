#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from tqdm import tqdm
from seqeval.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score,precision_recall_fscore_support,matthews_corrcoef
import numpy as np
# In[3]:


def train_fn(data_loader, model, optimizer, scheduler):
    model.train()
    torch.set_grad_enabled(True)
    total_train_loss = 0
    f1_scores_class_0=0
    f1_scores_class_1=0
    accuracy_scores = 0
    mcc_scores = 0
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

        loss = outputs[0]

        loss.backward()
        
        total_train_loss = total_train_loss + loss.item()
        
        labels = b_labels.view(-1)
        
        active_logits = outputs[1].view(-1, 2)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        
        active_loss = b_input_mask.view(-1) == 1
#         active_accuracy = labels.view(-1) != -100
        labels_tmp = torch.masked_select(labels, active_loss)
        pred_tmp = torch.masked_select(flattened_predictions, active_loss)
        lst_active_labels.extend(labels_tmp.tolist())
        lst_active_preds.extend(pred_tmp.tolist())
#         batch_f1_score_class_0 = f1_score(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy(),average='binary',pos_label = 0)
#         batch_f1_score_class_1 = f1_score(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy(),average='binary',pos_label = 1)
#         batch_accuracy_score = accuracy_score(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy())
#         batch_mcc_score = matthews_corrcoef(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy())
#         f1_scores_class0+=batch_f1_score_class_0 
#         f1_scores_class1+=batch_f1_score_class_1
#         accuracy_scores+=batch_accuracy_score
#         mcc_scores+=mcc_scores
        
        # Perform a backward pass to calculate the gradients.
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
#         break
#         print('Train loss:' ,total_train_loss)
    
    avg_f1_score_0=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 0)
    avg_f1_score_1=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 1)     
    avg_accuracy_score=accuracy_score(lst_active_labels,lst_active_preds)
    avg_mcc_score=matthews_corrcoef(lst_active_labels,lst_active_preds)
    
#     for label, pred in zip(lst_active_labels,lst_active_preds):
        
#         f1_score_class_0+=f1_score(label.cpu().numpy(),pred.cpu().numpy(),average='binary',pos_label = 0)
#         f1_score_class_1+=f1_score(label.cpu().numpy(),pred.cpu().numpy(),average='binary',pos_label = 1)     
#         accuracy_scores=accuracy_score(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy())
#         mcc_scores = matthews_corrcoef(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy())
        
#     avg_f1_score_0 = f1_scores_class_0/len(data_loader)
#     avg_f1_score_1 = f1_scores_class_1/len(data_loader)
#     avg_accuracy_score = accuracy_scores/len(data_loader)
#     avg_mcc_score = mcc_scores/len(data_loader) 
    print(f'Average TrainingF1 score for class 0 for current epoch is : {avg_f1_score_0}')
    print(f'Average Training F1 score for class 1 for current epoch is : {avg_f1_score_1}')
    print(f'Average Training Accuracy score for current epoch is : {avg_accuracy_score}')
    print(f'Average Training MCC score for class 0 for current epoch is : {avg_mcc_score}')
    
    return (float(total_train_loss / len(data_loader)),avg_f1_score_0,avg_f1_score_1, avg_accuracy_score, avg_mcc_score) 

# In[4]:


def eval_fn(data_loader, model):
    model.eval()
    total_val_loss = 0
    f1_scores_class_0=0
    f1_scores_class_1=0
    accuracy_scores = 0
    mcc_scores = 0
    eval_preds, eval_labels = [], []
    lst_active_labels = []
    lst_active_preds = []
        
    for i, batch in enumerate(tqdm(data_loader)):
        
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda()
        torch.set_grad_enabled(False)
        outputs = model(b_input_ids, 
                        attention_mask=b_input_mask,
                        labels=b_labels)

            # Get the loss from the outputs tuple: (loss, logits)
        loss = outputs[0]
        # Convert the loss from a torch tensor to a number.
        # Calculate the total loss.
        total_val_loss = total_val_loss + loss.item()
        
        labels = b_labels.view(-1)
        
        active_logits = outputs[1].view(-1, 2)# [batch_size * seq_len *2]
        flattened_predictions = torch.argmax(active_logits, axis=1)
        
        active_loss = b_input_mask.view(-1) == 1
#         active_accuracy = labels.view(-1) != -100
        labels_tmp = torch.masked_select(labels, active_loss)
        pred_tmp = torch.masked_select(flattened_predictions, active_loss)
        lst_active_labels.extend(labels_tmp.tolist())
        lst_active_preds.extend(pred_tmp.tolist())

        
        
#         print(labels_tmp.cpu().numpy())
#         print(pred_tmp.cpu().numpy())
#         eval_labels.extend(labels_tmp.cpu().numpy())
#         eval_preds.extend(pred_tmp.cpu().numpy())
        
#         batch_f1_score_class_0 = f1_score(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy(),average='binary',pos_label = 0)
#         batch_f1_score_class_1 = f1_score(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy(),average='binary',pos_label = 1)

# #         f1_scores.append(batch_f1_score)
# #         print(batch_f1_score)
#         batch_accuracy_score = accuracy_score(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy())
# #         accuracy_scores.append(batch_accuracy_score)
#         batch_mcc_score = matthews_corrcoef(labels_tmp.cpu().numpy(),pred_tmp.cpu().numpy())
# #         mcc_scores.append(batch_mcc_score)
#         f1_scores_class0+=batch_f1_score_class_0 
#         f1_scores_class1+=batch_f1_score_class_1
#         accuracy_scores+=batch_accuracy_score
#         mcc_scores+=mcc_scores
#         print(accuracy_scores,f1_scores,mcc_scores)
#         print('validation loss:' ,total_val_loss)


    avg_f1_score_0=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 0)
    avg_f1_score_1=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 1)     
    avg_accuracy_score=accuracy_score(lst_active_labels,lst_active_preds)
    avg_mcc_score=matthews_corrcoef(lst_active_labels,lst_active_preds)
    
#     for label, pred in zip(lst_active_labels,lst_active_preds):
        
#         f1_scores_class_0+=f1_score(label.cpu().numpy(),pred.cpu().numpy(),average='binary',pos_label = 0)
#         f1_scores_class_1+=f1_score(label.cpu().numpy(),pred.cpu().numpy(),average='binary',pos_label = 1)     
#         accuracy_scores=accuracy_score(label.cpu().numpy(),pred.cpu().numpy())
#         mcc_scores = matthews_corrcoef(label.cpu().numpy(),pred.cpu().numpy()) 
    
#     avg_f1_score_0 = f1_scores_class_0/len(data_loader)
#     avg_f1_score_1 = f1_scores_class_1/len(data_loader)
#     avg_accuracy_score = accuracy_scores/len(data_loader)
#     avg_mcc_score = mcc_scores/len(data_loader)
    print(f'Average validation F1 score for class 0 for current epoch is : {avg_f1_score_0}')
    print(f'Average validation F1 score for class 1 for current epoch is : {avg_f1_score_1}')
    print(f'Average validation Accuracy score for current epoch is : {avg_accuracy_score}')
    print(f'Average validation MCC score for class 0 for current epoch is : {avg_mcc_score}')
    
    return (float(total_val_loss / len(data_loader)) , avg_f1_score_0,avg_f1_score_1, avg_accuracy_score, avg_mcc_score) #eval_labels, eval_preds

# In[ ]:





# In[ ]:





# In[ ]:




