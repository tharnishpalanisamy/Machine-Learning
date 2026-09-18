import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import StandardScaler  , OneHotEncoder 
from sklearn.compose import ColumnTransformer  
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , confusion_matrix , roc_curve , roc_auc_score , precision_recall_curve , average_precision_score



df = pd.read_csv('./train.csv') 

df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()  

common_titles = set(['Mr' , 'Miss' , 'Mrs' , 'Master' , 'Dr' , 'Rev'] )

df['Title'] = df['Title'].apply(lambda x : x if x in common_titles else 'Rare' )

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 

features = ['Embarked' , 'Sex' , 'Age' , 'Pclass' , 'Fare' , 'FamilySize' ,'Title' ]

X = df[features]
y = df['Survived'] 



X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=10)


numerical_features = ['Age' , 'Pclass' , 'Fare' , 'FamilySize'] 
categorical_features = ['Sex' , 'Embarked' , 'Title' ] 

numerical_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy='median')) , 
    ('scaler' , StandardScaler() )
])

categorical_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy='most_frequent')) , 
    ('encoder' , OneHotEncoder(handle_unknown='ignore') )
]) 


preprocessing = ColumnTransformer([
    ('num' , numerical_pipeline , numerical_features) , 
    ('cat' , categorical_pipeline , categorical_features ) 
])

model = Pipeline([
    ('preprocessing' , preprocessing) , 
    ('model' , LogisticRegression() ) 
])

model.fit(X_train , y_train) 

probability = model.predict_proba(X_val)[:,1]

# print(probability)



thresholds = np.arange(0.01 , 1.00 , 0.02)   
best_threshold = 0.01 
bestscore = float('-inf')
for threshold in thresholds :  
    prediction = (probability >= threshold).astype(int) 

    score = f1_score(y_val , prediction) 
    if score > bestscore :
        bestscore = score 
        best_threshold = threshold  


    print('threshold : ' , threshold)
    print( 'Accuracy' , accuracy_score(y_val , prediction )) 

    print(f'Precision : {precision_score(y_val , prediction,zero_division=0):.4f}') 
    print(f'Recall : {recall_score(y_val , prediction,zero_division=0):.4f}')   

    
    print(f'f1 : {f1_score(y_val , prediction):.4f}')  


print('best' , best_threshold)

prediction = (probability >= best_threshold).astype(int) 

print('confusion Matrix : ' , confusion_matrix(y_val , prediction) )
print('threshold : ' , best_threshold) 
print( 'Accuracy' , accuracy_score(y_val , prediction )) 

print(f'Precision : {precision_score(y_val , prediction ,zero_division=0):.4f}') 
print(f'Recall : {recall_score(y_val , prediction ,zero_division=0):.4f}')   


print(f'f1 : {f1_score(y_val , prediction):.4f}') 


fpr , tpr  , thresholds = roc_curve(y_val, probability)   

plt.plot(fpr , tpr ) 
plt.xlabel('Fale positive Rate') 
plt.ylabel('True positive Rate')  
plt.title('Roc curve ' )

plt.show()  


print('ROC-AUC : ' , roc_auc_score(y_val , probability)) 


precision , recall , thresholds = precision_recall_curve(y_val , probability) 

plt.plot(recall , precision) 
plt.xlabel('recall') 
plt.ylabel('precision') 
plt.title('PR Curve') 
plt.show()

print('PR AUC : ' , average_precision_score(y_val , probability))