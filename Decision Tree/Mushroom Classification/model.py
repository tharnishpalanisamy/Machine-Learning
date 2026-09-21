import pandas as pd  
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.tree import DecisionTreeClassifier , export_text
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

df = pd.read_csv('./mushrooms.csv') 

X = df.drop('class' , axis=1)
y = df['class'] 

categorical_features = X.columns.tolist() 
print(categorical_features)


categorical_pipeline  = Pipeline([
    ('encoder' , OneHotEncoder(handle_unknown='ignore') ) 
])

preprocessor = ColumnTransformer([
    ('cat' , categorical_pipeline , categorical_features)
])


model = Pipeline([
    ('preprocessor' ,preprocessor ) , 
    ('model' , DecisionTreeClassifier(criterion='gini' ,  random_state=10))
])


X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=10, stratify=y) 

model.fit(X_train , y_train) 

pred = model.predict(X_val) 

train_pred = model.predict(X_train)

print(
    'Train Accuracy:',
    accuracy_score(y_train, train_pred)
)

print(
    'val Accuracy:',
    accuracy_score(y_val, pred)
)


tree = model.named_steps['model'] 

print('depth : ',tree.get_depth()) 
print('Leaves : ' , tree.get_n_leaves()) 


feature_names = model.named_steps['preprocessor'].get_feature_names_out() 

print(feature_names) 


import pandas as pd

importance = pd.Series(
    tree.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print(importance.head(15))



















# print('predictions : ' , pred)
# print('Correct Values : ' , y_val) 

# print('Accuracy : ' , accuracy_score(y_val , pred) )

# 1. Extract the feature names after OneHotEncoder transformation
feature_names = model.named_steps['preprocessor'].get_feature_names_out(categorical_features)

# 2. Extract the actual decision tree model from the pipeline
tree_model = model.named_steps['model']

# 3. Export the text using the correct model and feature names
tree_rules = export_text(tree_model, feature_names=list(feature_names))
print(tree_rules)


# train_accuracy = model.score(X_train, y_train)
# val_accuracy = model.score(X_val, y_val)

# print("Train accuracy:", train_accuracy)
# print("Validation accuracy:", val_accuracy)




def train_tree(depth):

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', DecisionTreeClassifier(
            criterion='gini',
            max_depth=depth,
            random_state=10
        ))
    ])

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    tree = model.named_steps['model']

    return (
        train_acc,
        val_acc,
        tree.get_depth(),
        tree.get_n_leaves()
    )


for depth in [1, 2, 3, 4, 5, 6, 7, 8, 10, None]:

    result = train_tree(depth)

    print(
        depth,
        "Train:", result[0],
        "Val:", result[1],
        "Actual Depth:", result[2],
        "Leaves:", result[3]
    )