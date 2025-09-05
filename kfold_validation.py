import random
import sys
from dataset.main import data
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from train import Trainer
import torch
import plot 
from model_use.main import choose_model
import numpy as np 

def k_fold_validation(k, num_people=23):
    patients = list(range(num_people))
    random.shuffle(patients)
    
    fold_size = num_people // k
    remainder = num_people % k
    kfold_patients = []
    start = 0
    
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0) 
        kfold_patients.append(patients[start:end])
        start = end
    
    return kfold_patients

def validate(model_name, emotion ,category, k , num_people= 23, subject_dependecy = 'subject_independent') : 
    pateints = k_fold_validation(k , num_people)
    len_patients = len(pateints)
    i = 0 
    if subject_dependecy == 'subject_independent' : 
        for test_person in pateints :
            print("a new procedure is takeing place . . . " ) 
            history = choose_model(model_name, emotion, category , test_person , fold_idx=i , subject_dependecy = 'subject_independent')
            if i ==0 : 
                train_loss = np.array(history['train_loss'])
                val_loss = np.array(history['val_loss'])
                train_acc = np.array(history['train_acc'])
                val_acc  = np.array(history['val_acc'])
            else : 
                train_loss += np.array(history['train_loss'])
                val_loss += np.array(history['val_loss'])
                train_acc += np.array(history['train_acc'])
                val_acc  += np.array(history['val_acc'])
            i+=1
        train_loss /= len_patients
        val_loss /= len_patients
        train_acc /= len_patients
        val_acc /= len_patients
        return train_loss , val_loss , train_acc , val_acc
    elif subject_dependecy == 'subject_dependent' : 
        results = choose_model(model_name, emotion, category , test_person = None , fold_idx=None , subject_dependecy = 'subject_independent')



def k_fold_data_segmentation(x, y, k):
    n = x.shape[0]
    idx = list(range(n))

    for i in range(k):
        start = int(i * n / k)
        end = int((i + 1) * n / k)

        idx_test = idx[start:end]
        idx_test_set = set(idx_test)
        idx_train = [j for j in idx if j not in idx_test_set]

        x_train = x[idx_train, :, :]
        y_train = y[idx_train]
        x_test = x[idx_test, :, :]
        y_test = y[idx_test]

        yield (x_train, x_test, y_train, y_test)
