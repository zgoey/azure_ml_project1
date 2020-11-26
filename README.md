# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about direct marketing success for a bank; we seek to predict a client will subscribe to a long term deposit (see also https://archive.ics.uci.edu/ml/datasets/bank+marketing).

The best performing model we have found using the Scitkit-learn and AutoML approaches is a logistic regression model with regularization strength of 1.7341 and a maximum of 5000 iterations. This model attains an accuracy of 91.97%.

## Scikit-learn Pipeline
The Hyperdrive/Scikit-learn approach uses a logistic regression classifier, where hyperparameter tuning is applied to two hyperarameters:
1. regularization strength
2. maximum number of iterations 

The dataset is first preprocessed by applying one-hot-encoding to the features "job", "contact" and "education" and simple encoding to other string-valued features, writing ones for true-like and zeros for false-like expression, respectively. It is then divided in a train and test set using a 80-20 split.

We use a lognormal parameter sampler for the regularization strength with a range between -5 and 3, which seems appropriate since we are more interested in varying the order of magnitude of this parameter as opposed to varyings its magnitude itself.  For the maximum number of iterations we use a choice sampler picking values from {50, 100, 500, 1000, 5000, 10000}. The choice sampler has the advantage that we can explicitly specify each of the values that we are interested in.

We choose to apply a bandit policy with an evaluation interval of 100 and a slack factor of 0.1 to ensure that we do not unnecesarily use our computing resources without making a significant progress in achieving better accuracies.

Our notebook shows the output of a hyperparameter run that finds a model with an accuracy of 91.97% using a egularization strength of 1.7341 and a maximum of 5000 iterations.


## AutoML
The best model found by AutoML is a soft voting ensemble based on a XGBoostClassifier. Please see the output of cell 10 in udacity-project.ipynb for detailed view on the hyperparameters used for this classifier. The voting ensemble attains an accuracy of 91.69%.

## Pipeline comparison
The logistic regression model used in the Scikit-learn pipeline is much simpler than the AutoML ensemble and also achieves a slightly better accuracy(91.97% vs 91.69%). The reason that AutoML does not find the logistic regression model is that it does not seem to be included in the list of models that were tried during the 30 minutes run. Besides that, the following other pipeline characteristics can cause differences:
- AutoML applies automatic featurization that is not included in the Scikit-learn piepeline
- AutoML does not necessarily the hyperparameter space in the way we specified for the Scikit-learn pipeline
- AutoML as use in our notebook assesses accuracy by applying a 5-fold cross validation as opposed to the simple train-test split applied in the Scikit-learn pipeline.

Of course, the slightly inferior accuracy and increased model complexity come with the not to understate advantage that AutoML  really is fully automatic. It does not require the specification of a model architecture and it also handles the choice of hyperparameters by itself.


## Future work
The maximum number of runs in the hyperparameter sampling step of the Scikit-learn pipeline can be increased to consider even more combination, which could imporve the accuracy even further. In addition to this the AutoML experiment can be configured to enable deep neural networks, allowing even more models to be considered, which also could also benefit the achieved accuracy.

