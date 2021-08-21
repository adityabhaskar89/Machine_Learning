import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
data  = load_breast_cancer()

#print(data.feature_names)
#print(data.target_names)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size= 0.2)

clf = KNeighborsClassifier(n_neighbors= 3)
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

#SVC- Support Vector Classifier

clf2 = SVC(kernel='linear', C=3)
clf2.fit(x_train, y_train)

print(clf2.score(x_test, y_test))

#DTC- Decision Tree Classifier

clf3 = DecisionTreeClassifier()
clf3.fit(x_train, y_train)

print(clf3.score(x_test, y_test))

#RFC- Random Forest Classifier

clf4 = RandomForestClassifier()
clf4.fit(x_train, y_train)

print(clf4.score(x_test, y_test))
