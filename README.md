# CVAE
Convolutional Variational-Autoencoder (CVAE) for anomaly detection in time series.
A fully unsupervised approach to anomaly detection based on Convolutional Neural Networks and Variational Autoencoders.

This is the code for a paper that has been accepted at the International Conference on Time Series and Forecasting, namely ITISE2019 (http://itise.ugr.es/). Soon the link to the paper whose name is 'Unsupervised Anomaly Detection in Time Series with Convolutional-VAE', authors Emanuele La Malfa (first_name.lastname <at> mail.polimi.it) and Gabriele La Malfa (please contact me directly/ if you need his mail).

## Requirements
These packages are required:
```
TensorFlow (>=1.14)
TensorFlow Probability (0.7.0)
Matplotlib (2.2.4, but it works also with older versions)
Pandas (0.24.2, the same as above)
Scipy (1.2.2, ...)
Numpy (1.16.4, ...)
```

## Hot to run
The software has been tested on:
- Windows 10, TensorFlow 1.14.0 with no GPU acceleration;
- Linux Ubuntu 19.04, TensorFlow-gpu-1.14.0, CUDA 10.0 and cuDNN 7.4.

Both the systems were provided with 'pip' package manager and everything has been installed with the command
'pip install <package_name>'.

Each dataset needs to be associated to a .json config file, to put into the 'config' folder.
Then just go to the folder 'cvae' and run a 'python experiments.py <cfg.json>'  where '<cfg.json>' is the path to the configuration file.
Example:
```
python experiments.py config/sin_config.json
```

## Hot to run on GPU
You need to install TensorFlow-gpu, CUDA and cuDNN. It is easy on Linux, harder on Windows: there are many setups possible, these experiments has been tested on Linux Ubuntu 19. Please note that this software requires CUDA 10.0 as TensorFlow Probability runs only with TF>=1.14.0 (https://www.tensorflow.org/install/ource#tested_build_configurations)

Just modify the config.json file at the entry 'device' with an existing GPU device, like for example '/device:GPU:0', then run the experiment as described in the section 'How to run'.

## Known Issues
If you face the issue 'ImportError: No module named tpu.tpu_config', try this solution:
```
sudo pip uninstall tensorflow_estimator
sudo pip install tensorflow_estimator
```

Sometimes it works like a piece of cake.
