"""
This script defines 4 different modes of traning and how they can be used with
GPU support in keras.
"""
#Standard imports
from keras.models import load_model
import importlib
from keras.utils import multi_gpu_model
import tensorflow as tf

#Custom imports
from src import config
from src.training.modeller import finetune_model

def training_scratch(optimiser_class, loss_class, metric_class):
    print("Training from scratch")

    optimizer = optimiser_class[config.optimiser][0](lr=config.learning_rate,
                               **optimiser_class[config.optimiser][1])
    loss = loss_class[config.loss]
    metric = metric_class[config.metric]

    if config.no_of_gpu > 1:
        print("Running in multi-gpu mode")
        with tf.device('/cpu:0'):
            build = getattr(importlib.import_module(config.model),"build")
            model = build(input_shape=(config.size, config.size, 1))

        gpu_model = multi_gpu_model(model, gpus = config.no_of_gpu)
        gpu_model.compile(loss= loss, optimizer=optimizer, metrics=[metric])
        model.compile(loss= loss, optimizer=optimizer, metrics=[metric])
    else:
        build = getattr(importlib.import_module(config.model),"build")
        model = build(input_shape=(config.size, config.size, 1))
        model.compile(loss= loss, optimizer=optimizer, metrics=[metric])
        gpu_model = None

    return model, gpu_model


def training_checkpoint():
    print("Training from prv checkpoint")

    if config.no_of_gpu > 1:
        print("Running in multi-gpu mode")
        with tf.device('/cpu:0'):
            model = load_model(config.model_path)

        gpu_model = multi_gpu_model(model, gpus = config.no_of_gpu)
        gpu_model.layers[-2].set_weights(model.get_weights())

    else:
        model = load_model(config.model_path)
        gpu_model = None

    return model, gpu_model


def fine_tune(optimiser_class, loss_class, metric_class):
    print("Fine tuning mode")

    optimizer = optimiser_class[config.optimiser][0](lr=config.learning_rate,
                               **optimiser_class[config.optimiser][1])
    loss = loss_class[config.loss]
    metric = metric_class[config.metric]

    if config.no_of_gpu > 1:
        print("Running in multi-gpu mode")
        with tf.device('/cpu:0'):
            build = getattr(importlib.import_module(config.model),"build")
            model = build(input_shape=(config.tile_size, config.tile_size, 3))
            model.load_weights(config.weights_path, by_name=True)

        gpu_model = multi_gpu_model(model, gpus = config.no_of_gpu)
        gpu_model.layers[-2].set_weights(model.get_weights())
        gpu_model.compile(loss= loss, optimizer=optimizer, metrics=[metric])
        model.compile(loss= loss, optimizer=optimizer, metrics=[metric])

    else:
        build = getattr(importlib.import_module(config.model),"build")
        model = build(input_shape=(config.tile_size, config.tile_size, 3))
        model.load_weights(config.weights_path, by_name=True)
        model.compile(loss= loss, optimizer=optimizer, metrics=[metric])
        gpu_model = None

    return model, gpu_model


def transfer_learning(optimiser_class, loss_class, metric_class):
    print("Transfer Learning mode")

    #build the model
    model_path = config.model_path
    gpu_model = load_model(model_path)

    build = getattr(importlib.import_module(config.model),"build")
    model = build(size = config.tile_size, chs = 3)
    model.set_weights(gpu_model.layers[-2].get_weights())

#    #freeze layers for transfer learning & load weights
    model = finetune_model(model)

    if config.no_of_gpu > 1:
        gpu_model = multi_gpu_model(model, gpus = config.no_of_gpu, cpu_relocation=True)
        print("Running in multi-gpu mode")
    else:
        gpu_model = None

# =============================================================================
#     #compile the model
#     gpu_model = compile_model(gpu_model, lr = config.learning_rate,
#                               optimiser = optimiser_class[config.optimiser],
#                               loss = loss_class[config.loss] ,
#                               metric = metric_class[config.metric])
# =============================================================================

        return model, gpu_model
