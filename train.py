
import os, sys
sys.path.append(os.path.abspath('./src/networks'))

#To handel OOM errors
import tensorflow as tf
from keras import backend as K
import keras.backend.tensorflow_backend as ktf
def get_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.9,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.clear_session()
ktf.set_session(get_session())

#Standard Imports
import pandas as pd
import numpy as np
import pickle
from keras.optimizers import Adam, RMSprop, Nadam, SGD

#Custom Imports
import config
from src.training import data_loader
from src.training.data_generator import DataGenerator
from src.training.keras_callbacks import get_callbacks
from src.training.training_modes import training_scratch, training_checkpoint, fine_tune, transfer_learning
from src.training.keras_history import generate_stats
from src.training.plots import save_plots

if __name__ == "__main__":

    base_path = config.base_path
    exp_name = config.exp_name

    #Params
        #Constants
    size = config.size
    classes = config.nclasses
    chs = config.chs

        #Training Params
    epochs = config.epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    initial_epoch = config.initial_epoch

    f = open(config.class_weights_path, 'rb')
    class_weights = pickle.load(f)

    training_frm_scratch = config.training_frm_scratch
    training_frm_chkpt = config.training_frm_chkpt
    fine_tuning = config.fine_tuning
    transfer_lr = config.transfer_lr
    trial = config.trial

    if sum((training_frm_scratch, training_frm_chkpt, fine_tuning, transfer_lr)) != 1:
        raise Exception("Conflicting training modes")


    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.build_source(base_path)

    if trial:
        print("Running in trail mode")
        samples = config.samples
        X_train =  X_train[:samples]
        y_train = y_train[:samples]
        X_val = X_val[:samples]
        y_val = y_val[:samples]
        X_test = X_test[:samples]
        y_test = y_test[:samples]

    train_spe = int(np.floor(len(X_train)/ batch_size)) #spe = Steps per epoch
    val_spe = int(np.floor(len(X_val)/batch_size))

    # Initialise training and validation generators
    train_generator = DataGenerator(base_path, file_paths =X_train, labels =y_train,
                                    batch_size = batch_size, dim=(size,size), n_channels=chs,
                                    n_classes= classes, shuffle=True)

    val_generator = DataGenerator(base_path, file_paths =X_val, labels =y_val,
                                         batch_size = batch_size, dim=(size,size), n_channels= chs,
                                         n_classes= classes, shuffle=True)

    loss_class = {'bin_cross': 'binary_crossentropy'}

    metric_class = {'acc':'acc'}

    optimiser_class = {'adam': (Adam, {}),
                   'nadam': (Nadam, {}),
                   'rmsprop': (RMSprop, {}),
                   'sgd':(SGD, {'decay':1e-6, 'momentum':0.90, 'nesterov':True})}


    if training_frm_scratch:
        model, gpu_model = training_scratch(optimiser_class, loss_class, metric_class)

    elif training_frm_chkpt:
        model, gpu_model = training_checkpoint()

    elif fine_tuning:
        model, gpu_model = fine_tune(optimiser_class, loss_class, metric_class)

    elif transfer_lr:
        model, gpu_model = transfer_learning(optimiser_class, loss_class, metric_class)


    print("Model training params:")
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    params = (trainable_count + non_trainable_count,trainable_count, non_trainable_count)

    print('Total params: {:,}'.format(params[0]))
    print('Trainable params: {:,}'.format(params[1]))
    print('Non-trainable params: {:,}'.format(params[2]))

    #Set callbacks
    callbacks_list = get_callbacks(model)

    # Start/resume training
    if config.no_of_gpu > 1:
        history = gpu_model.fit_generator(steps_per_epoch= train_spe,
                                          generator=train_generator,
                                          epochs=epochs,
                                          workers=4,
                                          use_multiprocessing=True,
                                          validation_data = val_generator,
                                          validation_steps = val_spe,
                                          initial_epoch = initial_epoch,
                                          class_weight = class_weights,
                                          callbacks = callbacks_list)

    else:
        history = model.fit_generator(steps_per_epoch= train_spe,
                                          generator=train_generator,
                                          epochs=epochs,
                                          workers=4,
                                          use_multiprocessing=True,
                                          validation_data = val_generator,
                                          validation_steps = val_spe,
                                          initial_epoch = initial_epoch,
                                          class_weight = class_weights,
                                          callbacks = callbacks_list)

    #Save final complete model
    filename = "model_ep_"+str(int(epochs))+"_batch_"+str(int(batch_size))
    model.save("./data/"+exp_name+"/"+filename+".h5")
    print("Saved complete model file at: ", filename+"_model"+".h5")

    #Save history
    history_to_save = generate_stats(history, config)
    pd.DataFrame(history_to_save).to_csv("./data/"+exp_name+"/"+filename + "_train_results.csv")
    save_plots(history, exp_name)
