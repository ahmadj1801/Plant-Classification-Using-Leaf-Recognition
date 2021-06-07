X = features.drop('label', axis=1)
Y = features['label']
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=0)
'''neural_classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
neural_classifier.fit(x_train, y_train)
classifications = neural_classifier.predict(x_test)'''
knn = KNeighborsClassifier(100, weights='distance')
knn.fit(x_train, y_train)
classifications = knn.predict(x_test)
print(classifications)
print(classification_report(y_test, classifications))
print(accuracy_score(y_test, classifications))