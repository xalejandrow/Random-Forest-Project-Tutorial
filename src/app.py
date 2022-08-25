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
#Preprocessing
#df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')
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
classif_grid=RandomForestClassifier(random_state=1107)
classif_grid_random=RandomizedSearchCV(estimator=classif_grid,n_iter=100,cv=5,random_state=1107,param_distributions=random_grid)
classif_grid_random.fit(X_train,y_train)
#save model 
final_model = '../models/final_model_rf.sav'
pickle.dump =(classif_grid_random, open(final_model, 'wb'))

