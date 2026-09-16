import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score

df = pd.read_csv('./train.csv') 


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 

df['IsAlone'] = (df['FamilySize'] == 1 ).astype(int) 

df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip() 

common_titles = set(['Mrs' , 'Mr' ,'Miss' , 'Master'])
df['Title'] = df['Title'].apply(lambda x : x if x in common_titles else 'Rare' ) 

features = ['Age' , 'Title' , 'Fare' , 'IsAlone' , 'Embarked' , 'Sex' , 'FamilySize']     

numeric_features = ['Age' , 'Fare' , 'IsAlone' , 'FamilySize' ]  
categorical_features = ['Title' , 'Embarked' , 'Sex' ] 

numeric_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy='median') ) , 
    ('scaler' , StandardScaler() )
])

categorical_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy='most_frequent')) , 
    ('encoder' , OneHotEncoder(handle_unknown='ignore')) 
])

preprocessing = ColumnTransformer([
    ('numeric' , numeric_pipeline , numeric_features) , 
    ('categorical' , categorical_pipeline , categorical_features) 
])

model = Pipeline([
    ('preprocessing' , preprocessing) , 
    ('model' , LogisticRegression() ) 
])


X = df[features] 
y = df['Survived'] 

X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=42 ) 

model.fit(X_train , y_train) 
prediction = model.predict(X_val) 

print( 'validation accuracy : ' , accuracy_score(y_val , prediction))  

# if model evaluation is good then we need to train on entire training data 
model.fit(X, y)

testing_data = pd.read_csv('./test.csv') 

testing_data['FamilySize'] = testing_data['SibSp'] + testing_data['Parch'] + 1 
testing_data['IsAlone'] = (testing_data['FamilySize'] == 1).astype(int) 
testing_data['Title'] = testing_data['Name'].str.split(',').str[1].str.split('.').str[0].str.strip() 
testing_data['Title'] = testing_data['Title'].apply(lambda x : x if x in common_titles else 'Rare' ) 


testingFeatures =  ['Age' , 'Title' , 'Fare' , 'IsAlone' , 'Embarked' , 'Sex' , 'FamilySize']  

testing = model.predict(testing_data[testingFeatures])

print(testing) 

submission = pd.DataFrame({
    'PassengerId' : testing_data['PassengerId'] , 
    'Name' : testing_data['Name'] , 
    'Survived' : testing
})

submission.to_csv('result.csv' , index=False)