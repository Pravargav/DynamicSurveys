import pandas as pd

df = pd.read_csv(r'C:\Users\dell\DynamicSurveys\surveyAPI-Flask\pythonProject\Disease_symptom_and_patient_profile_dataset.csv')
X = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]].values
y = df.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
X[:,1] = le.fit_transform(X[:,1])
X[:,2] = le.fit_transform(X[:,2])
X[:,3] = le.fit_transform(X[:,3])
X[:,5] = le.fit_transform(X[:,5])
X[:,6] = le.fit_transform(X[:,6])
X[:,7] = le.fit_transform(X[:,7])
X[:,8] = le.fit_transform(X[:,8])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', )
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)




from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': range(1, 31),
    'knn__metric': ['minkowski', 'euclidean', 'manhattan'],
    'knn__weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)


classifier = KNeighborsClassifier(metric='minkowski',n_neighbors=24,weights='distance')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


print(ac)
print(cm)
