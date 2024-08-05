import joblib

import pandas as pd
df = pd.read_csv(r'C:\Users\dell\surveyApi\pythonProject\SymptomDiesease.csv')


df = df[df.groupby('Disease')['Disease'].transform('size') >=10]
df.isna().sum()
df.loc[df.duplicated()]

df = df.drop_duplicates().reset_index(drop= True)
df.shape

dicc = {'Yes':1, 'No':0, 'Low':1, 'Normal':2, 'High':3, 'Positive':1, 'Negative':0, 'Male':0, 'Female': 1}
def replace(x, dicc= dicc):
    if x in dicc:
        x = dicc[x]
    return x
df = df.applymap(replace)

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

X = df.drop(['Disease'], axis= 1).values
y = df.Disease.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.4, shuffle= True, stratify= y, random_state=30)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size= 0.5, shuffle= True, stratify= y_val, random_state=30)

svc_pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight= 'balanced'))])
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])

svc_pipe.fit(X_train, y_train)
knn_pipe.fit(X_train, y_train)

ysvc_pred = svc_pipe.predict(X_val)
yknn_pred = knn_pipe.predict(X_val)

from sklearn.metrics import make_scorer
parameters = {
    'svc__C': [1],
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svc__gamma': ['scale', 'auto'],
    'svc__shrinking': [True, False],
}

grid_search = GridSearchCV(svc_pipe, parameters, cv=5)

grid_search.fit(X_train, y_train)

print("Best Score: ", grid_search.best_score_)
print("Best Params: ", grid_search.best_params_)

best_clf = grid_search.best_estimator_

import numpy as np
y_pred_test = best_clf.predict(X_test)
single_row = np.array([[1 ,0 , 1  ,0  ,23  ,0  ,2  ,2,0]])
y_pred_single = best_clf.predict(single_row)
print('Prediction for the single row: ', y_pred_single)


single_row = np.array([[1 ,1 , 1  ,0  ,154  ,1  ,2  ,3 ,1]])
joblib.dump(best_clf, r'C:\Users\dell\surveyApi\pythonProject\modelx\model2x.pkl')
modelMe = joblib.load( r'C:\Users\dell\surveyApi\pythonProject\modelx\model2x.pkl')
y_pred_single2 = modelMe.predict(single_row)
print('Prediction for the single row2: ', y_pred_single2[0])