from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    flip_y=0.15,
    random_state=10
)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = DecisionTreeClassifier(random_state=10) 

param_grid = {
    'criterion' : ['gini' , 'entropy'] , 
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split' : [5,10,15,20] , 
    'min_samples_leaf' : [1,5,10,20] 
}

grid = GridSearchCV(
    estimator=model , 
    param_grid=param_grid , 
    cv = 5 , 
    scoring='accuracy' , 
    n_jobs=1 
)


grid.fit(X_train , y_train) 


print("Best parameters:")
print(grid.best_params_)

print("\nBest CV score:")
print(grid.best_score_)



best_model = grid.best_estimator_

train_acc = best_model.score(X_train, y_train)
val_acc = best_model.score(X_val, y_val)

print("\nFinal Model")
print("Train:", train_acc)
print("Validation:", val_acc)

print("Depth:", best_model.get_depth())
print("Leaves:", best_model.get_n_leaves())



import pandas as pd

results = pd.DataFrame(grid.cv_results_)

cols = [
    "param_criterion",
    "param_max_depth",
    "param_min_samples_leaf",
    "mean_test_score",
    "std_test_score"
]

print(
    results[cols]
    .sort_values("mean_test_score", ascending=False)
    .head(10)
)