import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report 



df = pd.read_csv('./data/bank.csv') 


# #statistics
# print(df.head()) 
# print(df.shape) 

# print(df.info()) 
# print(df['deposit'].value_counts()) 

X = df.drop('deposit' , axis = 1 ) 
y = df['deposit'] 

numerical_features = ['age' , 'balance' , 'day' , 'duration' , 'campaign' , 'pdays' , 'previous' ] 

categorical_features = ['job' , 'marital' , 'education' , 'default' , 'housing' , 'loan' , 'contact' , 'month' , 'poutcome'] 


categorical_pipeline = Pipeline([
    ('encoder' , OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer([
    ('cat' , categorical_pipeline , categorical_features) , 
], remainder='passthrough') 


X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=10 )


# #model 
# results = {} 
# for i in [2, 10 , 50 , 100 , 150, 200 , 300 ] :  
#     model = Pipeline([
#         ('preprocessor' , preprocessor ) , 
#         ('classifier' , RandomForestClassifier(
#             n_estimators=i ,  
#             max_depth = 17, 
#             random_state=10 , 
#             max_features=None 
#         ) )
#     ])


#     model.fit(X_train , y_train) 

#     train_pred = model.predict(X_train) 
#     val_pred = model.predict(X_val) 

#     val_accuracy = accuracy_score(y_val , val_pred)
#     results[i] = val_accuracy
#     #accuracy 

#     print('training Accuracy : ' , accuracy_score(y_train , train_pred)) 
#     print('Validation Accuracy : ' , val_accuracy ) 


#     print('Confusion matrix : ' , confusion_matrix(y_val , val_pred)) 

#     print('Classification report : ' , classification_report(y_val , val_pred) ) 

#     print(model.named_steps["classifier"].feature_importances_)


# print(results) 

# best_paramter = 0 
# max_val = 0 

# for param , acc in results.items() :
#     if acc > max_val :
#         max_val = acc 
#         best_paramter = param 


# print(f"Best paramter : {best_paramter} | accuracy : {max_val}")




model = Pipeline([
        ('preprocessor' , preprocessor ) , 
        ('classifier' , RandomForestClassifier(
            n_estimators=100 ,  
            max_depth = 17, 
            random_state=10 , 
            max_features=None  ,  
            Bootstrap = True , 
            oob_score=True
        ) )
    ])

model.fit(X_train, y_train)



classifier = model.named_steps['classifier'] 
preprocessor_fitted = model.named_steps['preprocessor']  

feature_names = preprocessor_fitted.get_feature_names_out()
importance = classifier.feature_importances_ 

feature_importance = pd.DataFrame({
    'feature' : feature_names , 
    'importance' : importance
})

feature_importance = feature_importance.sort_values(
    'importance' , ascending=False 
)


feature_importance['original_feature'] = ""
for feature in numerical_features :
    mask = feature_importance['feature'] == f"remainder__{feature}"  
    feature_importance.loc[mask , 'original_feature' ] = feature  



for feature in categorical_features : 
    mask = feature_importance['feature'].str.startswith(f"cat__{feature}") 
    feature_importance.loc[mask , 'original_feature'] = feature 



grouped_importance = (feature_importance
                      .groupby('original_feature')['importance']
                      .sum() 
                      .sort_values(ascending=False) 

                      )

# print(grouped_importance)

print('OOB Score : ' , classifier.oob_score_)

val_pred = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, val_pred))