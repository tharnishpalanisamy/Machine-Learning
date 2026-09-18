import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import (
    accuracy_score , confusion_matrix , precision_score , 
    recall_score , roc_auc_score , 
    f1_score , roc_curve , 
    precision_recall_curve , 
    average_precision_score
    )

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
# prediction = model.predict(X_val) 
probability = model.predict_proba(X_val)[:,1] 

THRESHOLD = 0.5

# prediction = (probability >= THRESHOLD).astype(int) #changing threshold



# # print( 'validation accuracy : ' , accuracy_score(y_val , prediction))  

# # if model evaluation is good then we need to train on entire training data 


# # model.fit(X, y)

# # testing_data = pd.read_csv('./test.csv') 

# # testing_data['FamilySize'] = testing_data['SibSp'] + testing_data['Parch'] + 1 
# # testing_data['IsAlone'] = (testing_data['FamilySize'] == 1).astype(int) 
# # testing_data['Title'] = testing_data['Name'].str.split(',').str[1].str.split('.').str[0].str.strip() 
# # testing_data['Title'] = testing_data['Title'].apply(lambda x : x if x in common_titles else 'Rare' ) 


# # testingFeatures =  ['Age' , 'Title' , 'Fare' , 'IsAlone' , 'Embarked' , 'Sex' , 'FamilySize']  

# # testing = model.predict(testing_data[testingFeatures])

# # # print(testing) 

# # submission = pd.DataFrame({
# #     'PassengerId' : testing_data['PassengerId'] , 
# #     'Name' : testing_data['Name'] , 
# #     'Survived' : testing
# # })

# # submission.to_csv('result.csv' , index=False)  

# cm = confusion_matrix(y_val ,prediction ) 

# print('confusion matrix' ,  cm) 

# precision = precision_score(y_val , prediction) 
# recall = recall_score(y_val , prediction ) 
# roc = roc_auc_score(y_val , probability) 
# accuracy = accuracy_score(y_val , prediction)
# f1 = f1_score(y_val , prediction)

# print('accuracy : ' , accuracy) 
# print('precision : ' , precision) 
# print('recall : ' , recall) 
# print('Roc Curve  : ' , roc)
# print('f1 : ' , f1) 


# # fpr, tpr, thresholds = roc_curve(y_val, probability) 
# # print( 'Prob : ',probability) 
# # plt.plot(fpr , tpr) 
# # plt.plot([0, 1], [0, 1], linestyle="--")

# # print('Tpr : ' , tpr)
# # print('fpr : ' , fpr)
# # print('thresholds : ' , thresholds)



# # plt.xlabel('False postive rate') 
# # plt.ylabel('True postive rate') 
# # plt.title('Roc curve') 

# # plt.show()  

precision , recall , thresholds = precision_recall_curve(y_val , probability) 
precision, recall, thresholds = precision_recall_curve(
    y_val,
    probability
)

# print("Precision:", len(precision))
# print("Recall:", len(recall))
# print("Thresholds:", len(thresholds))
for i in range(10):
    print(
        f"Threshold: {thresholds[i]:.4f} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f}"
    )

plt.plot(  recall , precision ) 

plt.xlabel('Recall') 
plt.ylabel('Precision') 
plt.title('PR Curve') 

plt.show() 


pr_auc = average_precision_score(y_val , probability)

print('Score : ', pr_auc )


# thresholds = np.arange(0.01 , 1.00 , 0.01) 

# best_threshold = 0
# best_f1 = 0

# for threshold in thresholds:

#     y_pred = (probability >= threshold).astype(int)

#     precision = precision_score(y_val, y_pred)
#     recall = recall_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred)

#     print(
#         f"Threshold: {threshold:.2f} | "
#         f"Precision: {precision:.3f} | "
#         f"Recall: {recall:.3f} | "
#         f"F1: {f1:.3f}"
#     )

#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold

# print("\nBest threshold:", best_threshold)
# print("Best F1:", best_f1)