import pandas as pd  
import numpy as np 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.tree import DecisionTreeClassifier , export_text
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

df = pd.read_csv('./mushrooms.csv') 

X = df.drop('class' , axis=1)
y = df['class'] 

X['random_feature'] = np.random.randint(0 , 10000 , size=len(X)) 

categorical_features = [
    col for col in X.columns
    if col != "random_feature"
]


categorical_pipeline  = Pipeline([
    ('encoder' , OneHotEncoder(handle_unknown='ignore') ) 
])

preprocessor = ColumnTransformer([
    ('cat', categorical_pipeline, categorical_features),
    ('random', 'passthrough', ['random_feature'])
])


model = Pipeline([
    ('preprocessor' ,preprocessor ) , 
    ('model' , DecisionTreeClassifier(criterion='gini' , min_samples_split=100 ,  random_state=10))
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



importance = pd.Series(
    tree.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print(importance.head(15))



















# print('predictions : ' , pred)
# print('Correct Values : ' , y_val) 

# print('Accuracy : ' , accuracy_score(y_val , pred) )

# 1. Extract the feature names after OneHotEncoder transformation
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

# 2. Extract the actual decision tree model from the pipeline
tree_model = model.named_steps['model']

# 3. Export the text using the correct model and feature names
tree_rules = export_text(tree_model, feature_names=list(feature_names))
print(tree_rules)


# train_accuracy = model.score(X_train, y_train)
# val_accuracy = model.score(X_val, y_val)

# print("Train accuracy:", train_accuracy)
# print("Validation accuracy:", val_accuracy)




# def train_tree(depth):

#     model = Pipeline([
#         ('preprocessor', preprocessor),
#         ('model', DecisionTreeClassifier(
#             criterion='gini',
#             max_depth=depth,
#             random_state=10
#         ))
#     ])

#     model.fit(X_train, y_train)

#     train_pred = model.predict(X_train)
#     val_pred = model.predict(X_val)

#     train_acc = accuracy_score(y_train, train_pred)
#     val_acc = accuracy_score(y_val, val_pred)

#     tree = model.named_steps['model']

#     return (
#         train_acc,
#         val_acc,
#         tree.get_depth(),
#         tree.get_n_leaves()
#     )


# for depth in [1, 2, 3, 4, 5, 6, 7, 8, 10, None]:

#     result = train_tree(depth)

#     print(
#         depth,
#         "Train:", result[0],
#         "Val:", result[1],
#         "Actual Depth:", result[2],
#         "Leaves:", result[3]
#     )

# tree = model.named_steps["model"]

# print("Train:", accuracy_score(y_train, model.predict(X_train)))
# print("Val:", accuracy_score(y_val, model.predict(X_val)))
# print("Depth:", tree.get_depth())
# print("Leaves:", tree.get_n_leaves())


# for leaf in [1, 2, 5, 10, 20, 50, 100, 200]:

#     model = Pipeline([
#         ("preprocessor", preprocessor),
#         ("model", DecisionTreeClassifier(
#             criterion="gini",
#             min_samples_leaf=leaf,
#             random_state=10
#         ))
#     ])

#     model.fit(X_train, y_train)

#     train_acc = accuracy_score(
#         y_train,
#         model.predict(X_train)
#     )

#     val_acc = accuracy_score(
#         y_val,
#         model.predict(X_val)
#     )

#     tree = model.named_steps["model"]

#     print(
#         "min_samples_leaf:", leaf,
#         "| Train:", train_acc,
#         "| Val:", val_acc,
#         "| Depth:", tree.get_depth(),
#         "| Leaves:", tree.get_n_leaves()
#     )



model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(
        criterion="gini",
        random_state=10
    ))
])

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

print("Train Accuracy:",
      accuracy_score(y_train, train_pred))

print("Validation Accuracy:",
      accuracy_score(y_val, val_pred))

tree = model.named_steps["model"]

print("Depth:", tree.get_depth())
print("Leaves:", tree.get_n_leaves()) 















