# CVAE
Convolutional Variational-Autoencoder (CVAE) for anomaly detection in time series.
A fully unsupervised approach to anomaly detection based on Convolutional Neural Networks and Variational Autoencoders.

This is the code for a paper that has been accepted at the International Conference on Time Series and Forecasting, namely ITISE2019 (http://itise.ugr.es/). Soon the link to the paper whose name is 'Unsupervised Anomaly Detection in Time Series with Convolutional-VAE', authors Emanuele La Malfa (first_name.lastname <at> mail.polimi.it) and Gabriele La Malfa (please contact me directly/ if you need his mail).

## Requirements
TensorFlow (>=1.14)
TensorFlow Probability (0.7.0, requires TensorFlow>=1.14)
Matplotlib (2.2.4, but it works also with older versions)
Pandas (0.24.2, but the same as above)
Scipy (1.2.2, ...)
Numpy (1.16.4, ...)

## Hot to run
The software has been tested on:
- Windows 10, TensorFlow 1.14.0 with no GPU acceleration;
- Linux Ubuntu 19.04, TensorFlow-gpu-1.14.0, CUDA 10.0 and cuDNN 7.4.

Both the systems were provided with 'pip' package manager and everything has been installed with the command
'pip install <package_name>'.

Just go to the folder 'convolutional-vae/<experiment_name>/' and run a 'python experiments.py' to run on CPU.

## Hot to run on GPU
You need to install TensorFlow-gpu, CUDA and cuDNN. It is easy on Linux, harder on Windows: there are many setups possible, but please note that these experiments has been tested on Linux Ubuntu 19. Please note that this software requires CUDA 10.0 as TensorFlow Probability
runs only with TF>=1.14.0 (https://www.tensorflow.org/install/source#tested_build_configurations).

Just go to the folder 'convolutional-vae/<experiment_name>/gpu' and run a 'python experiments.py'.
