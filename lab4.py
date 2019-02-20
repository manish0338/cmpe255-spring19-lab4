from mnist import MNIST
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

mndata = MNIST('images')

X_train, y_train = mndata.load_training()

labels = []

for i in y_train.tolist():
    if(i%2==0):
        labels.append('Even')
    else:
        labels.append('Odd')
        
y_train = labels

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
knc_clf = KNeighborsClassifier()


print("Models are being trained ....")
sgd_clf.fit(X_train, y_train)
knc_clf.fit(X_train, y_train)

print("Training completed ....\n")

X_test, y_test = mndata.load_testing()

labels.clear()

for i in y_test.tolist():
    if(i%2==0):
        labels.append('Even')
    else:
        labels.append('Odd')
        
y_test = labels

print("Computing predictions .....")
y_pred_sgd = sgd_clf.predict(X_test)

print("SGDClassifier Accuracy: ", accuracy_score(y_test, y_pred_sgd))

print("\nComputing predictions .....")
y_pred_knc = knc_clf.predict(X_test)

print("KNeighborsClassifier Accuracy: ", accuracy_score(y_test, y_pred_knc))
