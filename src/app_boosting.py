from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import statsmodels.formula.api as smf
from sklearn.model_selection import RandomizedSearchCV



df_raw = pd.read_csv('../data/raw/titanic_train.csv')
df_transform=df_raw.drop(['Cabin','PassengerId', 'Ticket', 'Name'], axis=1)
df_transform['Embarked_clean']=df_transform['Embarked'].fillna('S')
df_transform['Age_clean']=df_transform['Age'].fillna(30)
df_transform['Sex_encoded']=df_transform['Sex'].apply(lambda x: 1 if x == 'female' else 0 )
df_transform['Embarked_S']=df_transform['Embarked_clean'].apply(lambda x: 1 if x == 'S' else 0 )    
df_transform['Embarked_C']=df_transform['Embarked_clean'].apply(lambda x: 1 if x == 'C' else 0 )
#Split
df=df_transform.copy()
X=df.drop(['Survived','Embarked', 'Sex', 'Embarked_clean', 'Age' ], axis=1)
y=df['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=70)
#Model
classifier=RandomForestClassifier(random_state=1107)
classifier.fit(X_train, y_train)
#Optimizacion hiperparametros
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
criterion=['gini','entropy']
random_grid = {'n_estimators': n_estimators,
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap,
'criterion':criterion}

xgb_2 = XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid_xgb = RandomizedSearchCV(xgb_2,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid_xgb.fit(X_train, y_train)

#save model to disk
final_model_2 = '../models/final_model_boost.sav'
pickle.dump =(xgb_2, open(final_model_2, 'wb'))

xgb_2 = grid_xgb.best_estimator_