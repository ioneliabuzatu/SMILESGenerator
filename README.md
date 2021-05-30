## SMILESGenerator - An LSTM based model to generate novel molecules

#### Main Dependencies
* Python >= 3.7
* experiment-buddy, [https://github.com/ministry-of-silly-code/experiment_buddy](https://github.com/ministry-of-silly-code/experiment_buddy)
* Torch

### Usage of this repo

1. First preprocess the data for the training with running
   ```python utils/preprocess_training_data.py```. This will create a file needed for th training under 
   `utils/resources/data/train_val_btaches.npz`. You can skip this step by downloading the preprocessed data from 
   [here](https://drive.google.com/file/d/1NxK0qCNYdVDi0bRVf5gstusjMnW1VZ6o/view?usp=sharing).
2. Then train model with ```python main.py```
3. Evaluate model according to the [FCD](https://github.com/bioinf-jku/FCD) metric with `metric_fcd.py`, gviven you 
   have a file `my_smiles.txt` - check `Notes` section below.

This code is based on the paper [Generating Focussed Molecule Libraries for Drug
Discovery with Recurrent Neural Networks, Segler et al, 2017](https://arxiv.org/pdf/1701.01329.pdf)

### Evaluation
After training evaluate the pretrained model with the file `generate_10k_smiles.py` - you will need to change the 
filepaths in `generate_smiles.py` accordingly.