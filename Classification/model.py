import pandas as pd 
from sklearn.model_selection import train_test_split 

df = pd.read_csv('./train.csv') 

# print(df.head())            #returns top 5 rows  
# print(df.shape)             #returns no of rows and cols 
# print(df.info())            #returns column , non-null count , datatype  
# print(df.describe())        #returns details(count , mean , std , min , max , 25% , 50% , 75%) of all columns 
# print(df.isnull().sum())    #returns null count of all cols 


# X = df.drop('Survived' , axis=1)  #axis = 1 column , 0 means rows 
# y = df['Survived'] 

# X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=42)  

# print(df["Survived"].value_counts())    #returns all the values in the column along with count

# print(df['Sex'].value_counts())

# print(df["Pclass"].value_counts())

# print(df["Embarked"].value_counts())

# print(df[['Age' , 'Fare' , 'Survived']].corr())  

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 

print(df['FamilySize'])

df['IsAlone'] = (df['FamilySize'] == 1 ).astype(int) 

print(df[["SibSp", "Parch", "FamilySize" , 'IsAlone']].head(10)) 

