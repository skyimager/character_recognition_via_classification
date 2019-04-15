# charni
CHAracter Recognition in Natural Images

- [Project Structure](#proj-struc)
- [Preparing dataset](#preparing-dataset)
- [Project description](#proj-des)
- [TO-DO](#to-do)
- [License](#license)


<a name="proj-struc"></a>
## 1. Project Structure

The project data and codes are arranged in the following manner:

```
charni
  ├── train.py
  ├── config.py
  ├── src  
  |   ├── evaluate.py
  |   ├── training/
  |   ├── evaluation/
  |   ├── networks/   
  |   └── utils/
  ├── data
  |   ├── English
  |         ├── GoodImg
  |         |    ├── ....
  |         |    ├── train.txt
  |         |    ├── test.txt
  |         |    ├── validation.txt
  |         |    ├── GoodImg.pkl
  |         ├── BadImg
  |         |    ├── ....
  |         |    ├── train.txt
  |         |    ├── test.txt
  |         |    ├── validation.txt
  |         |    ├── BadImg.pkl
  |  
```

_Data_: <br>
the `data` folder is not a part of this git project as it was heavy. The same can be downloaded from below link:

```sh
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
```

<a name="preparing-dataset"></a>
## 2. Preparing dataset
To enable dynamic loading of data with the help of Keras Generator and easy shuffling of data for different experiments, we prepare data scripts as `train.txt`, `validation.txt` and `test.txt`.

The present dataset is highly imbalanced (no_of_files_per_class has a max of 400+ and min of 30+). To handle this we can either do a downsampling of the dominant cases or oversampling of minority cases. Here we are doing oversampling of minority cases. The generated class weights are saved as a pickle file (`class_weights_maps.pkl`) and are to be used in keras while training. This is done using the script [here](./src/utils/test_train_split.py)

<a name="proj-des"></a>
## 3. Project description
The training script is `train.py` <br>

The entire training configuration including the dataset path, hyper-parameters and other arguments are specified in `config.py`, which you can modify and experiment with. It gives a one-shot view of the entire training process. <br>

The training can be conducted in 4 modes:
 - Training from scratch
 - Training from checkpoint
 - Fine tuning
 - Transfer Learning

Support for `parallel processing` at CPU level for datapreparation and subsequent `multi-gpu` training has been added.

Explore `config.py` to learn more about the parameters you can tweak.

A Notebook has been created for explaining the different training steps: <br>
Jupyter-notebook [LINK](./notebook/training.ipynb)

For training, make the desired configuration in `config.py` and then run `train.py` as:

```
python train.py
```

<a name="to-do"></a>
## 4. To-Do

- [x] Data download and Analysis
- [x] Data scripts preparation
- [x] Data loader
- [x] Data Generator
- [ ] Data Augmentor
- [x] Add multi-gpu support for training
- [ ] Networks
- [ ] Training and Parameter Tuning
