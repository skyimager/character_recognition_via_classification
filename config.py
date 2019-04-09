#!/usr/bin/env python3
# -*- coding: utf-8 -*-

base_path = "data/English/GoodImg"
exp_name = "first_exp"

size = 64
chs = 1
nclasses = 62
class_weights_path = "data/English/GoodImg/GoodImg.pkl"

no_of_gpu = 1
batch_size = 12*no_of_gpu #has to be even no.

epochs = 20
patience_lr = 5
factor_lr = 0.5
min_delta = 0.01
patience_es = 10  
  
training_frm_scratch = True
training_frm_chkpt = False
fine_tuning = False
transfer_lr = False
trial = False

    
if training_frm_scratch:
    model = 'cnn1' #used for importing from src networks
    initial_epoch = 0    
    optimiser = 'sgd'  #enter everything in small letters
    loss = 'bce_dice'
    metric = 'dice'    
    learning_rate = 0.001


if training_frm_chkpt:
    initial_epoch = 21 #training starts from 'initial epoch + 1'
    model_path = "data/inria_safenet_t2/checkpoint-21-0.31.h5"


if fine_tuning:
   model = "pspnet"
   weights_path="data/pretrained/pspnet/pspnet50_ade20k_n1.h5"

   initial_epoch = 0
    
   optimiser = 'sgd'  #enter everything in small letters
   loss = 'wbce_dice'
   metric = 'dice'    
   learning_rate = 0.0001  

    
if transfer_lr:     
    model_path = "data/inria_safenet_t2/checkpoint-20-0.32.h5" 
    model = "safenet"
    initial_epoch = 0  
    
    trainable_layers =  list(range(104)) #complete architecture for safenet

    optimiser = 'sgd'  #enter everything in small letters
    loss = 'bce_dice'
    metric = 'dice'    
    learning_rate = 0.0001    


if trial:
    print("Trial Mode Activated")
    epochs = 3
    no_of_samples = 10

