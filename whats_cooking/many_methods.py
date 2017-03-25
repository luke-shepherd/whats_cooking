# many_methods.py

# This file tests many different machine learning on the data, 
# primarily methods taken from the python package sklearn


import numpy as np
import sklearn.svm
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.neural_network
import sklearn.decomposition
import sklearn.tree

import dataParser

def accuracy(yhat, y):
    acc = 0.0
    for i in range(0, len(y)):
        act = y[i]
        pr = yhat[i]
        if act == pr: acc += 1.0
    return (acc / len(y))
         


# Read in data from json file and format into matrices
(classes, ingredients, X, y, y_cuisine, all_classes) = dataParser.parse_input('train.json')

# Split the data randomly into training and testing sets
(X_train, X_test, y_train, y_test, train_labels, test_labels) = dataParser.split_training_data(X, y, all_classes, y_cuisine, .75)



### Decision Tree ####

# Default sklearn decision tree
print 'Building Decision Tree...'
tr = sklearn.tree.DecisionTreeClassifier()
tr.fit(X_train, train_labels)

tree_yhat_train = tr.predict(X_train)
tree_yhat_test = tr.predict(X_test)
print 'Decision Tree training accuracy: ', accuracy(tree_yhat_train, train_labels)
print 'Decision Tree testing accuracy: ', accuracy(tree_yhat_test, test_labels)


### Singer-Crammer SVM ###

# A singer-crammer multiclass SVM is used
print 'Buiding Crammer Singer SVM...'
svm_classifier = sklearn.svm.LinearSVC(multi_class='crammer_singer', verbose=10)
svm_classifier.fit(X_train, train_labels)

svm_yhat_train = svm_classifier.predict(X_train)
svm_yhat_test = svm_classifier.predict(X_test)

print 'SingCram SVM training accuracy: ', accuracy(svm_yhat_train, train_labels)
print 'SingCram SVM testing accuracy: ', accuracy(svm_yhat_test, test_labels)


### KNN - N = 3 ###

# Distance is calculated as manhattan distance
print 'Preparing KNN...'
knn = sklearn.neighbors.KNeighborsClassifier(p=1, n_neighbors=3)
knn.fit(X_train, train_labels)
knn_yhat_test = knn.predict(X_test)

print 'KNN accuracy: ', accuracy(test_labels, knn_yhat_test)

### Naive Bayes ###
print 'Building Naive Bayes Net...'
nb = sklearn.naive_bayes.BernoulliNB()
nb.fit(X_train, train_labels)

nb_yhat_train = nb.predict(X_train)
nb_yhat_test = nb.predict(X_test)

print 'Naive Bayes training accuracy: ', accuracy(nb_yhat_train, train_labels)
print 'Naive Bayes testing accuracy: ', accuracy(nb_yhat_test, test_labels)

### SAG Logistic Regression ###

# use a very small iteration number otherwise it takes a very very long time
print 'Training Logistic Classifier...'
log_reg = sklearn.linear_model.LogisticRegression(max_iter=3, multi_class='multinomial', verbose=10, solver='sag')
log_reg.fit(X_train, train_labels)

log_yhat_train = log_reg.predict(X_train)
log_yhat_test = log_reg.predict(X_test)

print 'Logistic Regression training accuracy: ', accuracy(log_yhat_train, train_labels)
print 'Logistic Regression testing accuracy: ', accuracy(log_yhat_test, test_labels)

### MLP 100 hidden layers ###
print 'Training MLP...'
percept = sklearn.neural_network.MLPClassifier(tol=.1, solver='sgd', early_stopping=True, activation='tanh', alpha=0, 
                                                learning_rate='adaptive', max_iter=1000, warm_start=True)

percept.fit(X_train, train_labels)
mlp_yhat_train = percept.predict(X_train)
mlp_yhat_test = percept.predict(X_test)

print 'MLP training accuracy: ', accuracy(mlp_yhat_train, train_labels)
print 'MLP testing accuracy: ', accuracy(mlp_yhat_test, test_labels)


print 'DONE'
