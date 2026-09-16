import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import StandardScaler  , OneHotEncoder
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score 


df = pd.read_csv('./train.csv') 

# print(df.head())            #returns top 5 rows  
# print(df.shape)             #returns no of rows and cols 
# print(df.info())            #returns column , non-null count , datatype  
# print(df.describe())        #returns details(count , mean , std , min , max , 25% , 50% , 75%) of all columns 
# print(df.isnull().sum())    #returns null count of all cols 


# X = df.drop('Survived' , axis=1)  #axis = 1 column , 0 means rows 
# y = df['Survived'] 


# print(df["Survived"].value_counts())    #returns all the values in the column along with count

# print(df['Sex'].value_counts())

# print(df["Pclass"].value_counts())

# print(df["Embarked"].value_counts())

# print(df[['Age' , 'Fare' , 'Survived']].corr())  

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 

# print(df['FamilySize'])

df['IsAlone'] = (df['FamilySize'] == 1 ).astype(int) 

# print(df[["SibSp", "Parch", "FamilySize" , 'IsAlone']].head(10))  


df['Title'] = df['Name'].str.split(pat=',').str[1].str.split(pat='.').str[0].str.strip()
# print(df[['Title' , 'Name']].head(10)  )

# print(df['Title'].value_counts()) 

# print(df['Title'].unique())

common_titles = set(['Mr' , 'Mrs' , 'Miss' , 'Master']) 
# print(common_titles) 

df['Title'] = df['Title'].apply(lambda x : x if x in common_titles else 'Rare') 

# print(df['Title'].value_counts()) 

# print(df.describe())

# print(df['Age'].median())


numerical_features = ['Age' , 'FamilySize' , 'IsAlone' , 'Fare'] 
categorical_features= ['Title' , 'Sex' , 'Embarked']  
features = ['Age' , 'FamilySize' , 'IsAlone' , 'Fare', 'Title' , 'Sex' , 'Embarked'] 

features = [
    'Age',
    'FamilySize',
    'IsAlone',
    'Fare',
    'Title',
    'Sex',
    'Embarked'
]

X = df[features]
y = df['Survived'] 
X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=42)  



numerical_pipeline = Pipeline([
    ('imputer'  , SimpleImputer(strategy='median')  ) , 
    ('scalar' , StandardScaler() ) 
])


categorical_pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy='most_frequent')) , 
    ('encoder' , OneHotEncoder( handle_unknown='ignore' ) ) 
])

preprocessor = ColumnTransformer([
    ('num' , numerical_pipeline , numerical_features) ,
    ('cat' , categorical_pipeline , categorical_features) 
])


model_pipeline = Pipeline([
    ('preprocessor' , preprocessor ) , 
    ('model' , LogisticRegression()  ) 
])

model_pipeline.fit(X_train , y_train)  

predictions = model_pipeline.predict(X_val) 

print( 'Accuracy' , accuracy_score(y_val , predictions))

newPassanger = pd.DataFrame([{
    'Age' : 80 , 
    'Title' : 'Mr' , 
    'Sex' : 'Male' , 
    'FamilySize' : 2 , 
    'IsAlone' : 0 ,  
    'Fare' : 800 , 
    'Embarked' : 'S'
}])

newPred = model_pipeline.predict_proba(newPassanger) 

print('Sastha : ' , newPred  ) 