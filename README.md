# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset that we are investigating contains information about the success of direct marketing in a banking context. Based on various background characteristics, we seek to predict whether a client is susceptible to subscribe to a long-term deposit (see also https://archive.ics.uci.edu/ml/datasets/bank+marketing and https://core.ac.uk/download/pdf/55631291.pdf).

The best performing model we have found for this problem using the aforementioned approaches is a logistic regression model that was trained with a regularization strength of 0.4434 and a maximum of 10000 iterations. This model attains an accuracy of 91.85%.
## Scikit-learn Pipeline
The Hyperdrive/Scikit-learn approach uses a logistic regression classifier, where hyperparameter tuning is applied to two hyperparameters:
1. regularization strength
2. maximum number of iterations

The dataset is first preprocessed by applying one-hot-encoding to the features "job", "contact" and "education" and label encoding to the other string-valued features. It is then divided in a train and test set using a 80-20 split. 

RandomParameterSampling is used to sample the hyperparameter space. This has the benefit of reaching a reasonable solution in relatively little time. This solution can then later be refined in a full grid search, which takes up more time. We use the following parameter sampling distributions for each of the hyperparameters to be optimized:
- **lognormal parameter sampler** for the regularization strength with a range between -3 and 3 <br/>
- **choice sampler** for the maximum number of iterations sampling from {50, 100, 500, 1000, 5000, 10000}  <br/>

Using the aforementioned samplers and sampling ranges, we generate 100 hyperparameter combinations that are compared on their resulting classification accuracies on the test set.

To keep the computational burden in check, the hyperdrive run is configured with an early stopping policy called **bandit policy**, which uses an evaluation interval of 100 and a slack factor of 0.1. Runs that after 100 iterations need more than a 10% improvement to reach the current best accuracy are cancelled by this policy. The benefit of this is a more efficient usage of resources: the processing units are put to good use and are not wasting their time on computations that do not lead to a significant progress in achieving better accuracies.

The notebook udacity-project.ipynb shows the output of a hyperparameter run that results in a model with an accuracy of 91.85% using a regularization strength of 0.4434 and a maximum of 10000 iterations.


## AutoML
In the AutoML pipeline, the features first undergo the same encoding that was used in the scikit-learn pipeline. Classification performance is also measured by using accuracy as the primary metric, albeit that we now choose to apply a 5-fold cross validation, which helps to get a more stable accuracy estimate. The duration of the experiment is limited to 30 minutes to ensure that an answer will be reported in due time. 

AutoML tries out various pipelines, combining various feature scalers(MaxAbsScaler, StandardScalerWrapper, MinMaxScaler, RobustScaler) with various classifiers(XGBoostClassifier, LightGBM, SGD, RandomForest, ExtremeRandomTrees). The best model ultimately found by AutoML is a soft voting ensemble based on eight classifiers from previous AutoML runs: 

| Estimator                          | Hyperparameters Classifier                     | Weight      |
| -----------------------------------|------------------------------------------------|--------------
| MaxAbsScaler XGBoostClassifier     | base_score=0.5, booster='gbtree',              | 0.0667      |
|                                    | colsample_bylevel=1, colsample_bynode=1,       |             |
|                                    | colsample_bytree=1, gamma=0,                   |             |
|                                    | learning_rate=0.1, max_delta_step=0,           |             |
|                                    | max_depth=3, min_child_weight=1, missing=nan,  |             |
|                                    | n_estimators=100, n_jobs=1, nthread=None,      |             |
|                                    | objective='binary:logistic', random_state=0,   |             |
|                                    | reg_alpha=0, reg_lambda=1,                     |             |
|                                    | scale_pos_weight=1, seed=None, silent=None,    |             |
|                                    | subsample=1, tree_method='auto', verbose=-10,  |             |
|                                    | verbosity=0                                    |             |
|                                    |                                                |             |
| MaxAbsScaler LightGBM              | boosting_type='gbdt', class_weight=None,       | 0.5333      |
|                                    | colsample_bytree=1.0,                          |             |
|                                    | importance_type='split', learning_rate=0.1,    |             |
|                                    | max_depth=-1, min_child_samples=20,            |             |
|                                    | min_child_weight=0.001, min_split_gain=0.0,    |             |
|                                    | n_estimators=100, n_jobs=1, num_leaves=31,     |             |
|                                    | objective=None, random_state=None,             |             |
|                                    | reg_alpha=0.0, reg_lambda=0.0, silent=True,    |             |
|                                    | subsample=1.0, subsample_for_bin=200000,       |             |
|                                    | subsample_freq=0, verbose=-10                  |             |    
|                                    |                                                |             |                                     
| StandardScalerWrapper SGD          | alpha=7.346965306122448,                       | 0.0667      | 
|                                    | class_weight=None, eta0=0.001,                 |             |  
|                                    | fit_intercept=True,                            |             |  
|                                    | l1_ratio=0.8979591836734693,                   |             |  
|                                    | learning_rate='constant',                      |             |  
|                                    | loss='modified_huber', max_iter=1000,          |             |  
|                                    | n_jobs=1, penalty='none',                      |             |  
|                                    | power_t=0.6666666666666666,                    |             |  
|                                    | random_state=None, tol=0.01                    |             | 
|                                    |                                                |             |  
| StandardScalerWrapper SGD          | alpha=1.4286571428571428,                      | 0.0667      |
|                                    | class_weight=None, eta0=0.01,                  |             |
|                                    | fit_intercept=True,                            |             |
|                                    | l1_ratio=0.7551020408163265,                   |             |
|                                    | learning_rate='constant', loss='log',          |             |
|                                    | max_iter=1000, n_jobs=1, penalty='none',       |             |
|                                    | power_t=0.4444444444444444,                    |             |
|                                    | random_state=None, tol=0.001                   |             |  
|                                    |                                                |             |                            
| MinMaxScaler SGD                   | alpha=4.693930612244897,                       | 0.0667      |
|                                    | class_weight='balanced', eta0=0.001,           |             |
|                                    | fit_intercept=False,                           |             |
|                                    | l1_ratio=0.3877551020408163,                   |             |
|                                    | learning_rate='constant',                      |             |
|                                    | loss='squared_hinge', max_iter=1000,           |             |
|                                    | n_jobs=1, penalty='none',                      |             |
|                                    | power_t=0.3333333333333333,                    |             |
|                                    | random_state=None, tol=0.001                   |             |
|                                    |                                                |             |                                        
| StandardScalerWrapper SGD          | alpha=3.0612938775510203,                      | 0.0667      |
|                                    | class_weight='balanced', eta0=0.0001,          |             |    
|                                    | fit_intercept=True,                            |             |    
|                                    | l1_ratio=0.8979591836734693,                   |             |    
|                                    | learning_rate='constant',                      |             |    
|                                    | loss='modified_huber', max_iter=1000,          |             |    
|                                    | n_jobs=1, penalty='none',                      |             |    
|                                    | power_t=0.6666666666666666,                    |             |    
|                                    | random_state=None, tol=0.01                    |             |
|                                    |                                                |             | 
| StandardScalerWrapper RandomForest | bootstrap=False, ccp_alpha=0.0,                | 0.0667      |
|                                    | class_weight='balanced',                       |             |
|                                    | criterion='entropy', max_depth=None,           |             |
|                                    | max_features='sqrt',                           |             |
|                                    | max_leaf_nodes=None, max_samples=None,         |             |
|                                    | min_impurity_decrease=0.0,                     |             |
|                                    | min_impurity_split=None,                       |             |
|                                    | min_samples_leaf=0.035789473684210524,         |             |
|                                    | min_samples_split=0.15052631578947367,         |             |
|                                    | min_weight_fraction_leaf=0.0,                  |             |
|                                    | n_estimators=10, n_jobs=1,                     |             |
|                                    | oob_score=False, random_state=None,            |             |
|                                    | verbose=0, warm_start=False                    |             |                         
|                                    |                                                |             |  
| RobustScaler ExtremeRandomTrees    | bootstrap=False, ccp_alpha=0.0,                | 0.0667      |
|                                    | class_weight='balanced', criterion='gini',     |             | 
|                                    | max_depth=None, max_features='sqrt',           |             | 
|                                    | max_leaf_nodes=None, max_samples=None,         |             | 
|                                    | min_impurity_decrease=0.0,                     |             | 
|                                    | min_impurity_split=None,                       |             | 
|                                    | min_samples_leaf=0.06157894736842105,          |             | 
|                                    | min_samples_split=0.10368421052631578,         |             | 
|                                    | min_weight_fraction_leaf=0.0,                  |             | 
|                                    | n_estimators=50, n_jobs=1,                     |             | 
|                                    | oob_score=False, random_state=None,            |             | 
|                                    | verbose=0, warm_start=False                    |             |                            


## Pipeline comparison
The logistic regression model used in the Scikit-learn pipeline is much simpler than the AutoML ensemble and also achieves a slightly better accuracy (91.85% vs 91.66%). The reason that AutoML does not find the logistic regression model is that it does not seem to be included in the list of models that were tried during the 30 minutes run. Besides that, the following other pipeline characteristics can cause differences:
- AutoML applies automatic featurization that is not included in the Scikit-learn pipeline, 
- AutoML does not necessarily sample the hyperparameter space in the way we specified for the Scikit-learn pipeline, 
- AutoML as used in our notebook assesses accuracy by applying a 5-fold cross validation as opposed to the simple train-test split applied in the Scikit-learn pipeline.

Of course, the slightly lower accuracy and increased model complexity is countered by the fact that AutoML really is fully automatic. It does not require the specification of a model architecture, and it also handles the choice of hyperparameters by itself.


## Future work
The maximum number of runs in the hyperparameter sampling step of the Scikit-learn pipeline can be increased to consider more hyperparameter combinations, which could improve the accuracy even further. To get a more stable accuracy estimate, 5-fold cross validation could also be applied in the Scikit-learn pipeline, as it is in AutoML. The AutoML pipeline itself can be enhanced by enabling deep neural networks, which also could also lead to an increase in the achieved accuracy.


