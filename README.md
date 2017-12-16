# CGT Regression

The notebooks in this directory contain work building models to classify phenotype features from CGT representations of genotype.

## Contents

- LR_CGT_to_diagnosis.ipynb - notebook for binary classification logistic regression of diagnosis based on the genetic CGT matrix.
- LR_CGT_to_phenotype_cluster_k2.ipynb - notebook for binary classification logistic regression of phenotype cluster with k=2 based on the genetic CGT matrix.
- LR_CGT_to_phenotype_cluster_k3.ipynb - notebook for multi-class logistic regression classification of phenotype cluster with k=3 based on the genetic CGT matrix.
- class_EvalLR.py and .pyc - definition of EvalLR object, which automatically evaluates a logistic regression classifier with kfold cross validation. Essential for most noteboooks in this directory.
- class_SibKFold.py and .pyc - a script for partitioning data for cross-validation based on family membership

## TODO:

- [ ]  change relative paths so that jupyter scripts run in this directory


