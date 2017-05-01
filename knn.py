from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from scipy.spatial import distance

class KNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, x_test):
        predictions = []
        for item in x_test:
            label = self.closest(item)
            predictions.append(label)
        return predictions

    def closest(self, item):
        min_dist = distance.euclidean(item, self.X_train[0])
        min_ind = 0
        for i in range(1, len(self.X_train)):
            if (distance.euclidean(item, self.X_train[i]) < min_dist):
                min_dist = distance.euclidean(item, self.X_train[i])
                min_ind = i
            return self.Y_train[min_ind]

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

from sklearn.neighbors import KNeighborsClassifier

cls1 = KNeighborsClassifier()
cls1.fit(X_train, Y_train)

pred1 = cls1.predict(X_test)

cls2 = KNN()
cls2.fit(X_train, Y_train)
pred2 = cls2.predict(X_test)

from sklearn.metrics import accuracy_score

acc1 = accuracy_score(pred1, Y_test)
print '\n'
print '#########################################'
print '## Running classifiers in IRIS dataset ##'
print '#########################################'

print 'Acuuracy of KNN in scikit learn: ', acc1*100, '%'
acc2 = accuracy_score(pred2, Y_test)
print 'Acuuracy of KNN in our Implementation: ', acc2*100, '%'
