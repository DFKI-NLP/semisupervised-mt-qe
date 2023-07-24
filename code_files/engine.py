#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
from tqdm import tqdm
from seqeval.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef,f1_score,classification_report
import numpy as np
import config

# In[3]:


def train_fn(data_loader, model, optimizer, scheduler, large_model=False):
    model.train()
    total_train_loss = 0
    lst_active_labels = []
    lst_active_preds = []
    lst_active_src_labels=[]
    lst_active_src_labels_pred = []
    lst_active_gaps_labels = []
    lst_active_gaps_labels_pred = []
    lst_active_tar_labels =[]
    lst_active_tar_labels_pred = []
    lst_active_tar_and_gaps_labels =[]
    lst_active_tar_and_gaps_labels_pred = []
    for step , batch in enumerate(tqdm(data_loader, total = len(data_loader))):
        
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        _,outputs = model(b_input_ids, 
                        attention_mask=b_input_mask)

#         print(outputs[1].grad_fn)
        if large_model == True:
            loss = ClassificationCost(outputs,b_labels,b_input_mask).mean()
        else:
            loss = ClassificationCost(outputs,b_labels,b_input_mask)

        
        loss.backward()
        
        total_train_loss = total_train_loss + loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
       
        labels = b_labels.view(-1) 
        active_logits = outputs.view(-1, 2)
        flattened_predictions = torch.argmax(active_logits, axis=1)

        active_accuracy = labels.view(-1) != -100
        labels_tmp = torch.masked_select(labels, active_accuracy) 
        pred_tmp = torch.masked_select(flattened_predictions, active_accuracy) 
        lst_active_labels.extend(labels_tmp.tolist())
        lst_active_preds.extend(pred_tmp.tolist())
        
        # seperate source and target sentences labels in different lists (one extended and one for each example)

        src_labels , tar_labels, src_labels_example, tar_labels_example,input_ids_tar = div_src_tar(b_input_ids,b_labels) 
        
        # seperate gap labels and target labels into different lists, takes input the whole target label for each example in the batch
        
        gap_labels,tar_labels_nongaps,gap_labels_example, tar_labels_nongaps_example = get_gaps_labels(tar_labels_example,input_ids_tar)
        
        assert ( len(gap_labels_example[1]) + len(tar_labels_nongaps_example[1]) == len(tar_labels_example[1]))

        labels_pred_torch=predictions_per_sentence(flattened_predictions, config.TRAIN_BATCH_SIZE) # [...256],[..256],[...256]
        
        src_labels_pred, tar_labels_pred , src_labels_pred_example, tar_labels_pred_example,input_ids_tar_pred = div_src_tar(b_input_ids, labels_pred_torch)
        
        gap_labels_pred, tr_labels_pred,gap_labels_example_pred,tar_labels_nongaps_example_pred  = get_gaps_labels(tar_labels_pred_example,input_ids_tar_pred) 
        
        assert ( len(gap_labels_example_pred[1]) + len(tar_labels_nongaps_example_pred[1]) == len(tar_labels_pred_example[1]))
        
        assert(len(src_labels) == len(src_labels_pred))
        assert(len(tar_labels) == len(tar_labels_pred))
        assert(len(gap_labels) == len(gap_labels_pred))
        assert(len(tar_labels_nongaps) == len(tr_labels_pred))
        
        tensor_src_labels = torch.Tensor(src_labels)
        tensor_tar_labels = torch.Tensor(tar_labels)
        tensor_src_labels_pred = torch.Tensor(src_labels_pred)
        tensor_tar_labels_pred = torch.Tensor(tar_labels_pred)
        tensor_gap_labels = torch.Tensor(gap_labels)
        tensor_gap_labels_pred = torch.Tensor(gap_labels_pred)
        tensor_target_labels = torch.Tensor(tar_labels_nongaps)
        tensor_target_labels_pred = torch.Tensor(tr_labels_pred)
        
        active_accuracy = tensor_src_labels.view(-1) != -100
        src_labels_active = torch.masked_select(tensor_src_labels, active_accuracy) 
        src_labels_active_pred = torch.masked_select(tensor_src_labels_pred, active_accuracy)
        lst_active_src_labels.extend(src_labels_active.tolist())
        lst_active_src_labels_pred.extend(src_labels_active_pred.tolist())
        
        active_accuracy_gaps = tensor_gap_labels.view(-1) != -100
        gap_labels_active = torch.masked_select(tensor_gap_labels, active_accuracy_gaps) 
        gap_labels_active_pred = torch.masked_select(tensor_gap_labels_pred, active_accuracy_gaps)
        lst_active_gaps_labels.extend(gap_labels_active.tolist())
        lst_active_gaps_labels_pred.extend(gap_labels_active_pred.tolist())
        
        active_accuracy_tar = tensor_target_labels.view(-1) != -100 # excluding gaps
        target_labels_active = torch.masked_select(tensor_target_labels, active_accuracy_tar) 
        target_labels_active_pred = torch.masked_select(tensor_target_labels_pred, active_accuracy_tar)
        lst_active_tar_labels.extend(target_labels_active.tolist())
        lst_active_tar_labels_pred.extend(target_labels_active_pred.tolist())
        
        active_accuracy_tar_and_gaps = tensor_tar_labels.view(-1) != -100
        tar_labels_active = torch.masked_select(tensor_tar_labels, active_accuracy_tar_and_gaps) 
        tar_labels_active_pred = torch.masked_select(tensor_tar_labels_pred, active_accuracy_tar_and_gaps)
        lst_active_tar_and_gaps_labels.extend(tar_labels_active.tolist())
        lst_active_tar_and_gaps_labels_pred.extend(tar_labels_active_pred.tolist())
    

    avg_f1_score_0=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 0)
    avg_f1_score_1=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 1)     
    avg_accuracy_score=accuracy_score(lst_active_labels,lst_active_preds)
    avg_mcc_score=matthews_corrcoef(lst_active_labels,lst_active_preds)
    
    avg_f1_score_0_src=f1_score(lst_active_src_labels,lst_active_src_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_src=f1_score(lst_active_src_labels,lst_active_src_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_src=accuracy_score(lst_active_src_labels,lst_active_src_labels_pred)
    avg_mcc_score_src = matthews_corrcoef(lst_active_src_labels,lst_active_src_labels_pred)
    
    avg_f1_score_0_tar=f1_score(lst_active_tar_labels,lst_active_tar_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_tar=f1_score(lst_active_tar_labels,lst_active_tar_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_tar=accuracy_score(lst_active_tar_labels,lst_active_tar_labels_pred)
    avg_mcc_score_tar = matthews_corrcoef(lst_active_tar_labels,lst_active_tar_labels_pred)
    
    avg_f1_score_0_gaps=f1_score(lst_active_gaps_labels,lst_active_gaps_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_gaps=f1_score(lst_active_gaps_labels,lst_active_gaps_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_gaps=accuracy_score(lst_active_gaps_labels,lst_active_gaps_labels_pred)
    avg_mcc_score_gaps = matthews_corrcoef(lst_active_gaps_labels,lst_active_gaps_labels_pred)

    avg_f1_score_0_tar_and_gaps=f1_score(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_tar_and_gaps=f1_score(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_tar_and_gaps=accuracy_score(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred)
    avg_mcc_score_tar_and_gaps=matthews_corrcoef(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred)


    print('Average F1 Training score for class 0 :' ,avg_f1_score_0)
    print('Average F1 Training score for class 1 :' ,avg_f1_score_1)
    print('Average Accuracy Training score  :' ,     avg_accuracy_score)
    print('Average mcc Training score  :' ,avg_mcc_score)
    
    print('Average F1 Training score for source sentence class 0 :' ,avg_f1_score_0_src)
    print('Average F1 Training score for source sentence class 1 :' ,avg_f1_score_1_src)
    print('Average Accuracy Training score source sentence  :' ,avg_accuracy_score_src)
    print('Average mcc Training score source sentence :' ,avg_mcc_score_src)
    
    print('Average F1 Training score for target sentence class 0 :' ,avg_f1_score_0_tar)
    print('Average F1 Training score for target sentence class 1 :' ,avg_f1_score_1_tar)
    print('Average Accuracy Training score target sentence  :' ,avg_accuracy_score_tar)
    print('Average mcc Training score target sentence :' ,avg_mcc_score_tar)
    
    print('Average F1 Training score for gaps in target sentence class 0 :' ,avg_f1_score_0_gaps)
    print('Average F1 Training score for gaps in target sentence class 1 :' ,avg_f1_score_1_gaps)
    print('Average Accuracy Training score gaps in target sentence  :' ,avg_accuracy_score_gaps)
    print('Average mcc Training score gaps in target sentence :' ,avg_mcc_score_gaps)
    
    print('Average F1 Training score for whole target sentence class 0 :' ,avg_f1_score_0_tar_and_gaps)
    print('Average F1 Training score for whole target sentence class 1 :' ,avg_f1_score_1_tar_and_gaps)
    print('Average Accuracy Training score whole target sentence  :' ,avg_accuracy_score_tar_and_gaps)
    print('Average mcc Training score whole target sentence :' ,avg_mcc_score_tar_and_gaps)
    
    return (float(total_train_loss/len(data_loader)),avg_f1_score_0,avg_f1_score_1,avg_accuracy_score,
            avg_mcc_score,avg_f1_score_0_src,avg_f1_score_1_src,avg_accuracy_score_src,avg_mcc_score_src,
            avg_f1_score_0_tar,avg_f1_score_1_tar,avg_accuracy_score_tar,avg_mcc_score_tar,
            avg_f1_score_0_gaps,avg_f1_score_1_gaps,avg_accuracy_score_gaps,avg_mcc_score_gaps,
            avg_f1_score_0_tar_and_gaps,avg_f1_score_1_tar_and_gaps,avg_accuracy_score_tar_and_gaps,avg_mcc_score_tar_and_gaps,
            lst_active_labels,lst_active_preds,lst_active_src_labels,lst_active_src_labels_pred,
            lst_active_tar_labels,lst_active_tar_labels_pred,lst_active_gaps_labels,lst_active_gaps_labels_pred,
            lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred)


# In[4]:


def eval_fn(data_loader, model,large_model = False):
    model.eval()
    total_val_loss = 0
    lst_active_preds = []
    lst_active_labels = [] 
    lst_active_src_labels=[]
    lst_active_src_labels_pred = []
    lst_active_gaps_labels = []
    lst_active_gaps_labels_pred = []
    lst_active_tar_labels =[]
    lst_active_tar_labels_pred = []
    lst_active_tar_and_gaps_labels =[]
    lst_active_tar_and_gaps_labels_pred = []
    
#     for  batch in tqdm(data_loader,total=len(data_loader)):
    for  batch in data_loader:

        
        if (model.training!=True):
            
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_labels = batch[2].cuda()
    #         torch.set_grad_enabled(False)
            _ , outputs = model(b_input_ids, 
                            attention_mask=b_input_mask)

                # Get the loss from the outputs tuple: (loss, logits)
    #         loss = outputs[0]
            if large_model == True :
                loss = ClassificationCost(outputs,b_labels,b_input_mask).mean()
            else:
                loss = ClassificationCost(outputs,b_labels,b_input_mask)
            # Convert the loss from a torch tensor to a number.
            # Calculate the total loss.
            total_val_loss = total_val_loss + loss.item()
            labels = b_labels.view(-1)
            active_logits = outputs.view(-1,2)
            flattened_predictions = torch.argmax(active_logits, axis=1)
            active_loss = b_labels.view(-1) != -100
            labels_tmp = torch.masked_select(labels,active_loss)
            pred_tmp = torch.masked_select(flattened_predictions,active_loss)
            lst_active_labels.extend(labels_tmp.tolist())
            lst_active_preds.extend(pred_tmp.tolist())

            # seperate source and target sentences labels in different lists (one extended and one for each example)

            src_labels , tar_labels, src_labels_example, tar_labels_example,input_ids_tar = div_src_tar(b_input_ids,b_labels) 

            # seperate gap labels and target labels into different lists, takes input the whole target label for each example in the batch

            gap_labels,tar_labels_nongaps,gap_labels_example, tar_labels_nongaps_example = get_gaps_labels(tar_labels_example,input_ids_tar)

            assert ( len(gap_labels_example[0]) + len(tar_labels_nongaps_example[0]) == len(tar_labels_example[0]))

            labels_pred_torch=predictions_per_sentence(flattened_predictions, config.VALID_BATCH_SIZE) # [...256],[..256],[...256]

            src_labels_pred, tar_labels_pred , src_labels_pred_example, tar_labels_pred_example,input_ids_tar_pred = div_src_tar(b_input_ids, labels_pred_torch)

            gap_labels_pred, tr_labels_pred,gap_labels_example_pred,tar_labels_nongaps_example_pred  = get_gaps_labels(tar_labels_pred_example,input_ids_tar_pred) 

            assert ( len(gap_labels_example_pred[0]) + len(tar_labels_nongaps_example_pred[0]) == len(tar_labels_pred_example[0]))

            assert(len(src_labels) == len(src_labels_pred))
            assert(len(tar_labels) == len(tar_labels_pred))
            assert(len(gap_labels) == len(gap_labels_pred))
            assert(len(tar_labels_nongaps) == len(tr_labels_pred))

            tensor_src_labels = torch.Tensor(src_labels)
            tensor_tar_labels = torch.Tensor(tar_labels)
            tensor_src_labels_pred = torch.Tensor(src_labels_pred)
            tensor_tar_labels_pred = torch.Tensor(tar_labels_pred)
            tensor_gap_labels = torch.Tensor(gap_labels)
            tensor_gap_labels_pred = torch.Tensor(gap_labels_pred)
            tensor_target_labels = torch.Tensor(tar_labels_nongaps)
            tensor_target_labels_pred = torch.Tensor(tr_labels_pred)

            active_accuracy = tensor_src_labels.view(-1) != -100
            src_labels_active = torch.masked_select(tensor_src_labels, active_accuracy) 
            src_labels_active_pred = torch.masked_select(tensor_src_labels_pred, active_accuracy)
            lst_active_src_labels.extend(src_labels_active.tolist())
            lst_active_src_labels_pred.extend(src_labels_active_pred.tolist())

            active_accuracy_gaps = tensor_gap_labels.view(-1) != -100
            gap_labels_active = torch.masked_select(tensor_gap_labels, active_accuracy_gaps) 
            gap_labels_active_pred = torch.masked_select(tensor_gap_labels_pred, active_accuracy_gaps)
            lst_active_gaps_labels.extend(gap_labels_active.tolist())
            lst_active_gaps_labels_pred.extend(gap_labels_active_pred.tolist())

            active_accuracy_tar = tensor_target_labels.view(-1) != -100 # excluding gaps
            target_labels_active = torch.masked_select(tensor_target_labels, active_accuracy_tar) 
            target_labels_active_pred = torch.masked_select(tensor_target_labels_pred, active_accuracy_tar)
            lst_active_tar_labels.extend(target_labels_active.tolist())
            lst_active_tar_labels_pred.extend(target_labels_active_pred.tolist())

            active_accuracy_tar_and_gaps = tensor_tar_labels.view(-1) != -100
            tar_labels_active = torch.masked_select(tensor_tar_labels, active_accuracy_tar_and_gaps) 
            tar_labels_active_pred = torch.masked_select(tensor_tar_labels_pred, active_accuracy_tar_and_gaps)
            lst_active_tar_and_gaps_labels.extend(tar_labels_active.tolist())
            lst_active_tar_and_gaps_labels_pred.extend(tar_labels_active_pred.tolist())
        else:
            print('Model is in training mode while evaluation. Breaking!')
            break
    avg_f1_score_0=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 0)
    avg_f1_score_1=f1_score(lst_active_labels,lst_active_preds,average='binary',pos_label = 1)
    avg_accuracy_score=accuracy_score(lst_active_labels,lst_active_preds)
    avg_mcc_score = matthews_corrcoef(lst_active_labels,lst_active_preds)
    
    avg_f1_score_0_src=f1_score(lst_active_src_labels,lst_active_src_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_src=f1_score(lst_active_src_labels,lst_active_src_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_src=accuracy_score(lst_active_src_labels,lst_active_src_labels_pred)
    avg_mcc_score_src = matthews_corrcoef(lst_active_src_labels,lst_active_src_labels_pred)
    
    avg_f1_score_0_tar=f1_score(lst_active_tar_labels,lst_active_tar_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_tar=f1_score(lst_active_tar_labels,lst_active_tar_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_tar=accuracy_score(lst_active_tar_labels,lst_active_tar_labels_pred)
    avg_mcc_score_tar = matthews_corrcoef(lst_active_tar_labels,lst_active_tar_labels_pred)
    
    avg_f1_score_0_gaps=f1_score(lst_active_gaps_labels,lst_active_gaps_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_gaps=f1_score(lst_active_gaps_labels,lst_active_gaps_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_gaps=accuracy_score(lst_active_gaps_labels,lst_active_gaps_labels_pred)
    avg_mcc_score_gaps = matthews_corrcoef(lst_active_gaps_labels,lst_active_gaps_labels_pred)

    avg_f1_score_0_tar_and_gaps=f1_score(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred,average='binary',pos_label = 0)
    avg_f1_score_1_tar_and_gaps=f1_score(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred,average='binary',pos_label = 1)
    avg_accuracy_score_tar_and_gaps=accuracy_score(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred)
    avg_mcc_score_tar_and_gaps=matthews_corrcoef(lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred)

    # uncomment print during training
        
    print('Average F1 Validation score for whole sentence class 0 :' ,avg_f1_score_0)
    print('Average F1 Validation score for whole sentence class 1 :' ,avg_f1_score_1)
    print('Average Accuracy Validation score whole sentence  :' ,avg_accuracy_score)
    print('Average mcc Validation score whole sentence :' ,avg_mcc_score)

    print('Average F1 Validation score for source sentence class 0 :' ,avg_f1_score_0_src)
    print('Average F1 Validation score for source sentence class 1 :' ,avg_f1_score_1_src)
    print('Average Accuracy Validation score source sentence  :' ,avg_accuracy_score_src)
    print('Average mcc Validation score source sentence :' ,avg_mcc_score_src)
    
    print('Average F1 Validation score for target sentence class 0 :' ,avg_f1_score_0_tar)
    print('Average F1 Validation score for target sentence class 1 :' ,avg_f1_score_1_tar)
    print('Average Accuracy Validation score target sentence  :' ,avg_accuracy_score_tar)
    print('Average mcc Validation score target sentence :' ,avg_mcc_score_tar)
    
    print('Average F1 Validation score for gaps in target sentence class 0 :' ,avg_f1_score_0_gaps)
    print('Average F1 Validation score for gaps in target sentence class 1 :' ,avg_f1_score_1_gaps)
    print('Average Accuracy Validation score gaps in target sentence  :' ,avg_accuracy_score_gaps)
    print('Average mcc Validation score gaps in target sentence :' ,avg_mcc_score_gaps)
    
    print('Average F1 Validation score for whole target sentence class 0 :' ,avg_f1_score_0_tar_and_gaps)
    print('Average F1 Validation score for whole target sentence class 1 :' ,avg_f1_score_1_tar_and_gaps)
    print('Average Accuracy Validation score whole target sentence  :' ,avg_accuracy_score_tar_and_gaps)
    print('Average mcc Validation score whole target sentence :' ,avg_mcc_score_tar_and_gaps)
    
    return (float(total_val_loss/len(data_loader)),avg_f1_score_0,avg_f1_score_1,avg_accuracy_score,
            avg_mcc_score,avg_f1_score_0_src,avg_f1_score_1_src,avg_accuracy_score_src,avg_mcc_score_src,
            avg_f1_score_0_tar,avg_f1_score_1_tar,avg_accuracy_score_tar,avg_mcc_score_tar,
            avg_f1_score_0_gaps,avg_f1_score_1_gaps,avg_accuracy_score_gaps,avg_mcc_score_gaps,
            avg_f1_score_0_tar_and_gaps,avg_f1_score_1_tar_and_gaps,avg_accuracy_score_tar_and_gaps,avg_mcc_score_tar_and_gaps,
            lst_active_labels,lst_active_preds,lst_active_src_labels,lst_active_src_labels_pred,
            lst_active_tar_labels,lst_active_tar_labels_pred,lst_active_gaps_labels,lst_active_gaps_labels_pred,
            lst_active_tar_and_gaps_labels,lst_active_tar_and_gaps_labels_pred)
    


def div_src_tar(input_id, labels):
    
    src_labels=[]
    tar_labels=[]
    src_labels_example = []
    tar_labels_example = []
    index_labels=[]
    input_ids_tar = []
    input_id_np = input_id.cpu().numpy().copy()
    np_arr_labels = labels.cpu().numpy().copy()
    for items in input_id_np: # for each sentence in batch
      
        
        index_labels_tmp=np.where(items==2) # input id for seperator token{</s>} in xlm model , would be different for BERT {[SEP]}
        index_labels.append(index_labels_tmp[0]) # store indexes of (first)seperator in list
        
    for i,items in enumerate(index_labels):

        src_labels.extend(np_arr_labels[i][:items[1]])
        tar_labels.extend(np_arr_labels[i][items[1]:])
        src_labels_example.append(np_arr_labels[i][:items[0]])
        tar_labels_example.append(np_arr_labels[i][items[1]+1:])
        input_ids_tar.append(input_id[i][items[1]+1:])
        
        assert len(tar_labels_example) == len(input_ids_tar)
        assert len(input_ids_tar[0] == len(tar_labels_example[0]))
        assert(len(src_labels_example[i]) + len(tar_labels_example[i]) == config.MAX_LEN -2)

    return src_labels, tar_labels,src_labels_example,tar_labels_example,input_ids_tar
    
def predictions_per_sentence(flattened_predictions, batch_size):
                   
    # whole batch predictions to sentence-wise predictions
    labels_pred=[]
    assert len(flattened_predictions) % config.MAX_LEN == 0
    
    if len(flattened_predictions) != config.MAX_LEN * batch_size:
        len_rem_batches = int(len(flattened_predictions) / config.MAX_LEN)
        for i in range(len_rem_batches):
                   labels_pred.append(flattened_predictions[i*config.MAX_LEN:(i+1)*config.MAX_LEN])
    else:
        for i in range(batch_size):
            labels_pred.append(flattened_predictions[i*config.MAX_LEN:(i+1)*config.MAX_LEN])
            
    return torch.stack(labels_pred)

def get_gaps_labels(tar_labels,input_ids_tar):
    ''' Takes the input of list of target labels for all examples in the batch and returns list of gap labels and target labels '''
    
    gap_labels=[]
    gap_labels_example = []
    index_gap_labels=[]
    tar_labels_nongaps = []
    tar_labels_nongaps_example = []
    index_tar_labels = []
    input_id_np = np.array(input_ids_tar)
    np_arr_labels = np.array(tar_labels)

    
    for items in input_id_np:
        index_gap_labels_tmp = np.where(items.cpu() == 656) # 656 is the token id of the gap token that we are using in xlmroberta model  0.     ,y6
        index_gap_labels.append(index_gap_labels_tmp)
        
        index_tar_labels_tmp = np.where(items.cpu()!=656)
        index_tar_labels.append(index_tar_labels_tmp)

    for i,items in enumerate(index_gap_labels):
        gap_labels.extend(np_arr_labels[i][items])
        gap_labels_example.append(np_arr_labels[i][items])
    
    for i,items in enumerate(index_tar_labels):
        tar_labels_nongaps.extend(np_arr_labels[i][items])
        tar_labels_nongaps_example.append(np_arr_labels[i][items])

    return gap_labels,tar_labels_nongaps, gap_labels_example, tar_labels_nongaps_example

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
        class_0_weights = 1

    try:
        class_1_weights =1/len(torch.where(active_labels==1)[0])
    
    except:
        class_1_weights = 1
    
    weights_tensor = torch.tensor([class_0_weights,class_1_weights]).cuda()
    lfn = nn.CrossEntropyLoss(weight = weights_tensor)
    loss = lfn(active_logits,active_labels)
    
    return loss





# In[ ]:





# In[ ]:




