from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

import functions as f


def switch_bagging(argument):

    switcher = {
        1: (data1, target1, fscore1, (10, 0.225, 0.345)),
        2: (data2, target2, fscore2, (4, 0.125, 0.4)),
        3: (data3, target3, fscore3, (34, 0.6, 0.225)),
        4: (data4, target4, fscore4, (33, 0.675, 0.325)),
    }
    return switcher.get(argument)

def switch_boosting(argument):

    switcher = {
        1: (data1, target1, fscore1, (22, 0.05)),
        2: (data2, target2, fscore2, (12, 0.15)),
        3: (data3, target3, fscore3, (51, 0.25)),
        4: (data4, target4, fscore4, (71, 0.05)),
    }
    return switcher.get(argument)

def switch_random_forest(argument):

    switcher = {
        1: (data1, target1, fscore1, (1, 0.225, 17, 0.225)),
        2: (data2, target2, fscore2, (27, 0.175, 25, 0.125)),
        3: (data3, target3, fscore3, (48, 0.175, 33, 0.125)),
        4: (data4, target4, fscore4, (41, 0.875, 11, 0.125)),
    }
    return switcher.get(argument)


fscore1 = []
fscore2 = []
fscore3 = []
fscore4 = []
filename1 = 'wine.data.csv'
filename2 = 'diabetes.data.csv'
filename3 = 'glass.data.csv'
filename4 = 'abalone_improved.data.csv'
dataset1 = f.import_file(filename1)
dataset2 = f.import_file(filename2)
dataset3 = f.import_file(filename3)
dataset4 = f.import_file(filename4)


data1, target1 = f.break_class_wine(dataset1)
data2, target2 = f.break_class_other(dataset2)
data3, target3 = f.break_class_other(dataset3)
data4, target4 = f.break_class_wine(dataset4)
data1 = np.array(data1)
'''for i in range(0, 5):
    for j in range (0, 13):
        k = np.float(data1[i][j])
        print(k)
    print("\n")'''
data2 = np.array(data2)
data3 = np.array(data3)
data4 = np.array(data4)
#  data1 = preprocessing.normalize(data1, axis=0, norm='l2')
'''for i in range(0, 5):
    for j in range (0, 13):
        k = np.float(data1[i][j])
        print("%.2f" % k)
    print("\n")'''
#  data2 = preprocessing.normalize(data2, axis=0, norm='l2')
#  data3 = preprocessing.normalize(data3, axis=0, norm='l2')
#  data4 = preprocessing.normalize(data4, axis=0, norm='l2')
target1 = np.array(target1)
target2 = np.array(target2)
target3 = np.array(target3)
target4 = np.array(target4)

gnb = GaussianNB()

'''
--------------------------------------------------BADANIE-A-------------------------------------------------------------

data, target, _= switch_boosting(4)

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in (skf.split(data,target)):
    X_train = data[train_index]
    X_test = data[test_index]
    y_train, y_test = target[train_index], target[test_index]


# clf = BaggingClassifier(gnb)
clf = AdaBoostClassifier(gnb)
# clf = RandomForestClassifier(gnb, max_depth=2, random_state=0)

clf.fit(X_train,y_train)
predicted = clf.predict(X_test)
#print('predicted', np.shape(predicted), type(predicted), predicted)
#print('labels', np.shape(y_test), type(y_test), y_test)
type_of_labels = np.unique(target)

f = f1_score(y_test, predicted, average='macro', labels=type_of_labels)
print("Accuracy:  ", accuracy_score(y_test, predicted)*100, "%")
print("Precision: ", precision_score(y_test, predicted, average='macro', labels=type_of_labels)*100, "%")
print("Recall:    ", recall_score(y_test, predicted, average='macro', labels=type_of_labels)*100, "%")
print("F1 score:  ", f*100, "%")


------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------BADANIE-B-------------------------------------------------------------
'''
#tab = np.arange(1, 50)
tab = np.arange(0.125, 1, 0.05)

for j in [1, 2, 3, 4]:

    gnb = GaussianNB()
    data, target, fscore_tab, (best_n_estimators, best_max_samples, best_max_features) = switch_bagging(j)
    #  data, target, fscore_tab, (best_n_estimators, best_learning_rate) = switch_boosting(j)
    #  data, target, fscore_tab, (best_n_estimators, best_max_features, best_max_depth, best_min_samples_split) = switch_random_forest(j)
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in (skf.split(data, target)):
        X_train = data[train_index]
        X_test = data[test_index]
        y_train, y_test = target[train_index], target[test_index]

    for i in tab:
        clf = BaggingClassifier(gnb, n_estimators=best_n_estimators, max_samples=best_max_samples, max_features=best_max_features)
        # clf = AdaBoostClassifier(gnb, n_estimators=best_n_estimators, learning_rate=best_learning_rate)
        #clf = RandomForestClassifier(n_estimators=best_n_estimators, max_features=best_max_features, max_depth=best_max_depth, min_samples_split=i)

        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        type_of_labels = np.unique(target)

        f = f1_score(y_test, predicted, average='micro', labels=type_of_labels)
        fscore_tab.append(f)

print("Best test for Wines:", np.argmax(fscore1)+1, "with value of f-score: ", fscore1[np.argmax(fscore1)])
print("Best test for Diabetes:", np.argmax(fscore2)+1, "with value of f-score: ", fscore2[np.argmax(fscore2)])
print("Best test for Glass:", np.argmax(fscore3)+1, "with value of f-score: ", fscore3[np.argmax(fscore3)])
print("Best test for Abalone:", np.argmax(fscore4)+1, "with value of f-score: ", fscore4[np.argmax(fscore4)])


# plotowanie wykresów

plt.figure(1)

plt.title('Minimum number of samples required to split a node')
plt.xlabel('N.N.')
plt.ylabel('F-score')
plt.plot(tab, fscore1, 'o--', label="Wines")
plt.plot(tab, fscore2, 'o--', label="Diabetes")
plt.plot(tab, fscore3, 'o--', label="Glass")
plt.plot(tab, fscore4, 'o--', label="Abalone")
plt.legend()
plt.show()
