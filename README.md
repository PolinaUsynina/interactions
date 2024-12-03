# Probing Earth’s ionosphere with pulsar polarimetry and deep learning

## **Annotation:**
In this project, we are solving the problem of self-calibration of pulsar polarimetry data in order to clean them from the influence of the ionosphere. We modernize the standard thin layer model of ionosphere by considering its dependence on solar and geomagnetic parameters using ML and DL. In this repository you can find the code for creating a dataset of solar and geomagnetic parameters for the selected pulsar as well as code for experiments with ML models and weights of trained neural networks.\
**The main results of the work were obtained on three pulsars: J0332+5434, J1921+2153 and J0814+7429, but here you will find the data only for J0332+5434. The data for two other pulsars can be provided upon reasonable request.**\
To install requirements run:
```
 !pip install -r requirements.txt
```
## Usage instruction for NOTEBOOKS folder
### get_features.ipynb
In this notebook you can find the code that will create a raw dataset of solar and geomagnetic parameters for the selected pulsar.

**How to use**:
1. Download all the data. You may change the ID of files if you need another pulsar
2. Launch and follow the notebook
3. Be careful, it may take some time to download and compute the dataset (up to 60 minutes)
4. Download the file obtained as a result of the notebook execution to your device

### preprocessing.ipynb
In this notebook you can find the code that performs the preprocessing of pulsar data, solar and geomagnetic parameters for chosen pulsar. As a result of executing the code, you will get a dataset ready for experiments.

**How to use**:
1. Download all the data. You may change the ID of files if you need another pulsar
2. Launch and follow the notebook
3. Feel free to change parameters in the section with instability experiments
4. Download the file obtained as a result of the notebook execution to your device

### ML_models.ipynb
In this notebook you can find the code for experiments with ML models

**How to use**:
1. Download all the data. You may change the ID of files if you need another pulsar
2. Launch and follow the notebook
3. Feel free to change hyperparameters of models if needed

## Usage instruction for interaction.ipynb
This is the main notebook to run the code for neural network experiments. If you want to **re-train the model with your hyperparameters**, you need to:
1. Run the cells from the beginning up to `from train import load_config, train`at the first launch
2. Set up hyperparametes in **configs/train.yaml** file. You can change parameters such as:
   - `max_epochs`
   - `batch_size`
   - `optimizer` and its `optimizer_params`. Built-in optimizers are listed in `pl_models.py` file
   - `scheduler` and its `scheduler_params`. Built-in optimizers are listed in `pl_models.py` file
4. Please choose unique `exp_name` for your experiment in **configs/train.yaml** file
5. Run all the cells from `train_config = load_config('configs/train.yaml')` up to `train(train_config)`
6. Your model will be saved in the **Experiments/neural_iono/** in the subfolder with the selected experiment name

If you want to **run the inference of an existing model**, you need to:
1. Choose the model from **Experiments/neural_iono**. The model which was trained on the training part of the pulsar J0332+5434 is located in the subfolder **J0332_train_val_test**. Its weitghts can be found in file **model.ckpt**
2. Write the name of chosen model in corresponding paths in **configs/inference.yaml** file
3. Please change the path in cell `t1_predicted = np.loadtxt("/home/jupyter/datasphere/project/interactions/Experiments/neural_iono/model_J0332/val/predictions.txt")`
4. Run the cells from `from inference import load_config, inference` up to `get_quality(RM_real, RM_predicted, test, show_stat = True)`
