# FreqRISE
This repository implements the methods presented in "FreqRISE: Explaining time series models using frequency masking" to appear at the Northern Lights Deep Learning conference 2025 (https://openreview.net/forum?id=JBH3mtjG9I#discussion).

Requirements:
- python=3.12.8
- h5py=3.12.1
- lightning=2.4.0
- torch=2.5.1
- zennit=0.5.1

## Reproducing the results
To reproduce the results from the paper, you need to download the AudioMNIST dataset by cloning this repository: https://github.com/soerenab/AudioMNIST and running the preprocessing script. 

We have uploaded the weights for the models used in the paper, which are located in the models folder. However, to reproduce the model training run the following steps.
### Model training
To train the models for AudioMNIST run:
```
python3 train_audiomnist.py --model_path /path/to/save/model --data_path /path/to/AudioMNIST/ --labeltype labeltype
```
data_path should point to the top folder which contains the folder "preprocessed_data" created through the preprocessing. Labeltype should be either digit or gender.

To train the models for the synthetic data run:
```
python3 train_synthetic.py --model_path /path/to/save/model --noise_level noise
```
noise_level refers to the amount of noise added to the training data. In the paper, we test a low noise setting with a noise level of 0.01 and a high noise setting with 0.8.

### Computing attributions
To compute the attributions run the script main_attributions.py. This computes attributions for both integrated gradients, LRP and FreqRISE.

If you find the code useful, please consider citing:
```
@inproceedings{
      br{\"u}sch2024freqrise,
      title={Freq{RISE}: Explaining time series using frequency masking},
      author={Thea Br{\"u}sch and Kristoffer Knutsen Wickstr{\o}m and Mikkel N. Schmidt and Tommy Sonne Alstr{\o}m and Robert Jenssen},
      booktitle={Northern Lights Deep Learning Conference 2025},
      year={2024},
      url={https://openreview.net/forum?id=JBH3mtjG9I}
}
```
