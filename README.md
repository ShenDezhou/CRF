# CRF
A Conditional Random Field Model based Chinese Word Segmentation Project.

## Introduction
CRF model takes advantage of contextual information, thus compared to HMM model, CRF improves the accuracy and recall.

## Supported Keras Loss and Optimizer
Loss:
1. supports loss function:"mean_squared_error","mean_absolute_error","mean_absolute_percentage_error","mean_squared_logarithmic_error","squared_hinge","hinge","categorical_crossentropy","sparse_categorical_crossentropy","binary_crossentropy","kullback_leibler_divergence","poisson","cosine_proximity"
2. NOT support function:"categorical_hinge","logcosh"
3. Training error function:"sparse_categorical_crossentropy",
Optimizer:
1. supports optimizer function:'sgd','rmsprop','adadelta','adam','adamax','nadam'
2. NOT support function:'adagrad',
3. NOT support serialization: 'tfoptimizer'.

