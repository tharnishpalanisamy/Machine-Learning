from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score 



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

model1 = DecisionTreeClassifier(criterion='gini' , random_state=10 ) 

model2 = DecisionTreeClassifier(criterion='entropy' , random_state=10)  


model1.fit(X_train , y_train) 

gini_train = model1.predict(X_train) 
gini_val = model1.predict(X_val) 

print("Gini Train:", accuracy_score(y_train, gini_train))
print("Gini Val:", accuracy_score(y_val, gini_val))


model2.fit(X_train , y_train) 

entropy_train = model2.predict(X_train) 
entropy_val = model2.predict(X_val) 

print('Entropy Train : ' , accuracy_score(y_train , entropy_train)) 
print('Entropy val : ' , accuracy_score(y_val , entropy_val)) 



print("\nGini")
print("Depth:", model1.get_depth())
print("Leaves:", model1.get_n_leaves())

print("\nEntropy")
print("Depth:", model2.get_depth())
print("Leaves:", model2.get_n_leaves())

for leaf in [1, 2, 5, 10, 20]:

    model = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=leaf,
        random_state=10
    )

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    print(
        f"min_samples_leaf={leaf} | "
        f"Train={train_acc:.3f} | "
        f"Val={val_acc:.3f} | "
        f"Depth={model.get_depth()} | "
        f"Leaves={model.get_n_leaves()}"
    )