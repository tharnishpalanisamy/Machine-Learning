import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score , classification_report 

df = pd.read_csv('./data/bank.csv') 

X = df.drop('deposit' , axis=1) 
y = df['deposit'] 

numerical_features = ['age' , 'balance' , 'day' , 'duration' , 'campaign' , 'pdays' , 'previous' ] 

categorical_features = ['job' , 'marital' , 'education' , 'default' , 'housing' , 'loan' , 'contact' , 'month' , 'poutcome'] 

categorical_pipeline = Pipeline([
    ('encoder'  , OneHotEncoder(handle_unknown='ignore') )
])

preprocessor = ColumnTransformer([
    ('cat' , categorical_pipeline , categorical_features) , 
    ('num' , 'passthrough' , numerical_features) 
])

params_grid = {
    'classifier__n_estimators' : [ 10, 50 , 100 , 150] , 
    'classifier__max_depth' : [2,5,10,15,17, 20 ] , 
    'classifier__max_features' : [ 'log2' , None] ,  
}

model = Pipeline([
    ('preprocessor' , preprocessor) , 
    ('classifier' , RandomForestClassifier(random_state=10 , bootstrap=True  ))
])


X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=10 ) 

grid = GridSearchCV(model , param_grid=params_grid , cv = 5 , scoring='accuracy' , n_jobs=-1 , verbose=1) 

grid.fit(X_train , y_train ) 

print('Best Params : ' , grid.best_params_) 

print("best CV score : " , grid.best_score_ ) 

best_model = grid.best_estimator_ 
val_pred = best_model.predict(X_val) 

print("\nValidation Accuracy:")
print(accuracy_score(y_val, val_pred))

print("\nClassification Report:")
print(classification_report(y_val, val_pred))


