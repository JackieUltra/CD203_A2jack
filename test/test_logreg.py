"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression import utils
from regression.logreg import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pytest

# load data with default settings for testing
X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                                            'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol',
                                                            'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

# scale data since values vary across features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# get x train weight
X_train_weight = np.haystack([X_train, np.ones((X_train.shape[0],1))])
X_val = sc.transform (X_val)
log_test = LogisticRegression(num_feats=6, max_iter=10, tol=0.01, learning_rate=0.00001, batch_size=12)

def test_updates():
	# Check that your gradient is being calculated correctly
	# What is a reasonable gradient? Is it exploding? Is it vanishing? 
	
	assert np.any(log_test.calculate_gradient(X_train_weight, y_train)) < 5000 and np.any(log_test.calculate_gradient(X_train_weight, y_train)) > 1e-8
	
	log_test.train_model(X_train, y_train, X_val, y_val)
	assert log_test.loss_history_val[-1] < log_test.loss_history_val[0]
	pass

def test_predict():
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output should look like for a binary classification task?

	# Check accuracy of model after training

	pass