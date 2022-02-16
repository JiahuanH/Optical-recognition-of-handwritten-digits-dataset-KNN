import collections
import time
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
x = digits.data
y = digits.target
# F1: print the dataset
print(x.shape)
print(y.shape)
samples, features = x.shape
print('The number of total data entries:', samples)
print('Class number:', len(digits.target_names))
print('Class labels:', np.unique(y))

entriesData = collections.Counter(digits.target)
print('The data entries is', entriesData)
# print the maximum and the minimum feature
print('Max feature', digits.data.max())
print('Min feature', digits.data.min())
print("========================================")

# F2: split training set and testing set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)
# KNN classifier
kNN_classifier = KNeighborsClassifier(n_neighbors=3)
kNN_classifier.fit(X_train, y_train)
y_predict = kNN_classifier.predict(X_test)
Y_predict = kNN_classifier.predict(X_train)

# F4(1): the train error and test error of the this model
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy of test data:%f' % accuracy)
print('Misclassified sample of test: %d' % (y_test != y_predict).sum())
print('Error of test data:%f' % (1 - accuracy))

accuracy1 = accuracy_score(y_train, Y_predict)
print('Accuracy of train data:%f' % accuracy1)
print('Misclassified sample of Train: %d' % (y_train != Y_predict).sum())
print('Error of train data:%f' % (1 - accuracy1))
print("========================================")


# F3: Implement an algorithm that can train the model
class MyKNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        result = []
        for j in X:
            distance = np.sqrt(np.sum((j - self.X) ** 2, axis=1))
            index = distance.argsort()
            index = index[:self.k]
            count = np.bincount(self.y[index])
            result.append(count.argmax())
        return np.asarray(result)


def TrainMyKNN():
    myKnn_start = time.time()
    knn = MyKNN(k=3)
    knn.fit(X_train, y_train)
    y_predict1 = knn.predict(X_test)
    Y_predict1 = knn.predict(X_train)

    # F4(2): the train error and test error of the this my knn model
    accuracy2 = accuracy_score(y_test, y_predict1)
    print('Accuracy of test data:%f' % accuracy2)
    print('Misclassified sample of test: %d' % (y_test != y_predict1).sum())
    print('Error of test data:%f' % (1 - accuracy2))

    accuracy3 = accuracy_score(y_train, Y_predict)
    print('Accuracy of train data:%f' % accuracy3)
    print('Misclassified sample of Train: %d' % (y_train != Y_predict1).sum())
    print('Error of train data:%f' % (1 - accuracy3))
    print('========================================')
    myKnn_end = time.time()
    useTime = myKnn_end - myKnn_start
    print('MyKnn model uses time:', useTime)
    print('========================================')


print(TrainMyKNN())


# F5: Allow users to query the models by changing the input
def ChangingInput():
    # choose the model
    command = input('Input A or B to choose the model(A:sklearn-KNN, B:myKNN)\n')
    X, Y = "A", "B"
    for counter in range(0, 100):
        if str(command) == X:

            command2 = input('Input the number which you want to train in range of 360\n')
            Predicted = KNeighborsClassifier(n_neighbors=3)
            Predicted.fit(X_train, y_train)
            # testData= model1.fit(X_train,y_train);
            PredictData = Predicted.predict(X_train)
            print("The predict results is")
            print(PredictData[int(command2)])
            print("The actual results is")
            print(y_test[int(command2)])

            break
        elif str(command) == Y:
            command2 = input('Input the number which you want to train in range of 360\n')
            knn = MyKNN(k=3)
            knn.fit(X_train, y_train)
            testData = knn.predict(X_train)
            testData2 = knn.predict(X_test)
            print("The predict results is")
            print(testData[int(command2)])
            print("The actual results is")
            print(testData2[int(command2)])
            break

        else:
            print("Please follow the Hint")
            counter += 1


print(ChangingInput())
