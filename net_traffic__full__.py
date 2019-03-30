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

# Load dataset
url = "../Datasets/network_traffic/traffic.csv"
names = ['Avg Packet Size', 'Std. deviation of packet size', 'Total Bytes', 'AVG. IAT(ms)', 'Std. daviation of IAT(ms)', 'Flow duration(SECONDS)', 'Push_flag_cnt', 'Avg_wind_size', 'class']
dataset = pandas.read_csv(url, names=names)

N_FEATURES = 8
N_CLASS = 2

# shape
print(dataset.shape)

# Peek at the Data (head)
#print(dataset.head(20))

# Statistical Summary
#print(dataset.describe())

# class distribution
#print(dataset.groupby('class').size())


"""
Data Visualization
""\"
# Univariate Plots 
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()

# Multivariate Plots
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
"""

# Split-out validation dataset
array = dataset.values
X = array[:,0:N_FEATURES]
Y = array[:,N_FEATURES]
# We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.
validation_size = 0.20
seed = 7
# We are using the metric of 'accuracy' to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.
scoring = 'accuracy'

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Spot Check Algorithms
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
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

"""
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))