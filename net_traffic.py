"""
Data used for this script: 
https://raw.githubusercontent.com/networkedsystemsIITB/Traffic_Classification/master/scripts/v1.0/KNN-classification-2-interface-machine/data.csv
"""

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

"""
Step 1: Prepare dataset
"""
# Load dataset
url = "../Datasets/network_traffic/data.csv"
dataset = pandas.read_csv(url)
dataset.set_value(dataset['Class']=='download',['Class'],0)
dataset.set_value(dataset['Class']=='multimedia',['Class'],1)
dataset = dataset.apply(pandas.to_numeric)

# shape
print(dataset.shape)

N_FEATURES = 8
N_CLASS = 2

# Split-out validation dataset
array = dataset.values
X = array[:,0:N_FEATURES]
Y = array[:,N_FEATURES]
seed = 7
# We are using the metric of 'accuracy' to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.
scoring = 'accuracy'
# 20% as test
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=seed)


"""
Spot Check Algorithms

This is just for inspecting which classifier could possibly work best with our data
"""
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))

"""
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""

"""
Step 2: Train the classifier

The result above shows that Decision Tree algorithm performs best
"""
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)


"""
Step 3: Test the classifier
"""
predictions = classifier.predict(X_validation)

print("\nTest accuration: {}".format(accuracy_score(Y_validation, predictions)))
print "Prediction :"
print np.asarray(predictions, dtype="int32")
print "Target :"
print np.asarray(Y_validation, dtype="int32")
print(classification_report(Y_validation, predictions))