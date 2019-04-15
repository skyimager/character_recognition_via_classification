#!/usr/bin/env python3
# -*- coding: utf-8 -*-

base_path = "data/English/GoodImg"
exp_name = "charni_squeezenet"

size = 64
chs = 3
nclasses = 62
class_weights_path = "data/English/GoodImg/GoodImg.pkl"

no_of_gpu = 2
batch_size = 64*no_of_gpu #has to be even no.

epochs = 30
patience_lr = 10
factor_lr = 0.5
min_delta = 0.01
patience_es = 10

training_frm_scratch = False
training_frm_chkpt = False
fine_tuning = False
transfer_lr = True
trial = False

if trial:
    print("Trial Mode Activated")
    epochs = 3
    samples = 200


if training_frm_scratch:
    model = 'squeezenet' #used for importing from src networks
    initial_epoch = 0
    optimiser = 'adam'  #enter everything in small letters
    loss = 'cat_cross'
    metric = 'acc'
    learning_rate = 0.001


if training_frm_chkpt:
    initial_epoch = 21 #training starts from 'initial epoch + 1'
    model_path = "data/inria_safenet_t2/checkpoint-21-0.31.h5"


if fine_tuning:
   model = "pspnet"
   weights_path="data/pretrained/pspnet/pspnet50_ade20k_n1.h5"

   initial_epoch = 0

   optimiser = 'sgd'  #enter everything in small letters
   loss = 'cat_cross'
   metric = 'acc'
   learning_rate = 0.0001


if transfer_lr:
    weights_path = "data/pretrained/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model = "squeezenet"
    initial_epoch = 0

    freeze_layers =  list(range(39)) #complete architecture for squeezenet

    optimiser = 'sgd'  #enter everything in small letters
    loss = 'cat_cross'
    metric = 'acc'
    learning_rate = 0.0001
