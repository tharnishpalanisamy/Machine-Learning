# from sklearn.datasets import make_classification 
# from sklearn.model_selection import train_test_split 
# from sklearn.tree import DecisionTreeClassifier 
# from sklearn.metrics import accuracy_score 



# X, y = make_classification(
#     n_samples=1000,
#     n_features=20,
#     n_informative=2,
#     n_redundant=0,
#     n_repeated=0,
#     n_classes=2,
#     flip_y=0.15,
#     random_state=10
# )

# X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2 , random_state=10 , stratify=y) 

# model = DecisionTreeClassifier(criterion='gini' , random_state=10) 

# model.fit(X_train , y_train) 

# train_pred = model.predict(X_train)
# val_pred = model.predict(X_val)

# print("Train:", accuracy_score(y_train, train_pred))
# print("Validation:", accuracy_score(y_val, val_pred))

# print("Depth:", model.get_depth())
# print("Leaves:", model.get_n_leaves())


# depths  = [1, 2, 3, 4, 5, 6, 8, 10, None]  
# min_split = [2,10,50,100, 150 , 200, 250 , 500 , 650]  
# min_leaf = [30, 50 , 100 , 160 , 200 , 250 , 300 , 400 , 500 ]

# # for i in range(9):  
# #     for j in range(9) :
# #         for k in range(9) : 
# #             depth = depths[i]  
# #             leaf = min_leaf[j] 
# #             sample = min_split[k]
# #             model = DecisionTreeClassifier(
# #                 criterion="gini",
# #                 max_depth=depth, 
# #                 min_samples_leaf= leaf , 
# #                 min_samples_split= sample , 
# #                 random_state=10
# #             )
        
# #             model.fit(X_train, y_train)
        
# #             train_acc = accuracy_score(
# #                 y_train,
# #                 model.predict(X_train)
# #             )
        
# #             val_acc = accuracy_score(
# #                 y_val,
# #                 model.predict(X_val)
# #             )
        
# #             print(
# #                 "Depth:", depth,
# #                 "| Min sample : " , sample ,
# #                 "| Min leaft : " , leaf ,
# #                 "| Train:", train_acc,
# #                 "| Val:", val_acc,
# #                 "| Actual depth:", model.get_depth(),
# #                 "| Leaves:", model.get_n_leaves()
# #             )



# # for i in range(9):  
# #     depth = depths[i]  
# #     leaf = 40
# #     sample = 80
# #     model = DecisionTreeClassifier(
# #         criterion="gini",
# #         max_depth=depth, 
# #         min_samples_leaf= leaf , 
# #         min_samples_split= sample , 
# #         random_state=10
# #     )

# #     model.fit(X_train, y_train)

# #     train_acc = accuracy_score(
# #         y_train,
# #         model.predict(X_train)
# #     )

# #     val_acc = accuracy_score(
# #         y_val,
# #         model.predict(X_val)
# #     )

# #     print(
# #         "Depth:", depth,
# #         "| Min sample : " , sample ,
# #         "| Min leaft : " , leaf ,
# #         "| Train:", train_acc,
# #         "| Val:", val_acc,
# #         "| Actual depth:", model.get_depth(),
# #         "| Leaves:", model.get_n_leaves()
# #     )


# # for split in [2, 5, 10, 20, 50, 100, 200]:

# #     model = DecisionTreeClassifier(
# #         criterion="gini",
# #         min_samples_split=split,
# #         random_state=10
# #     )

# #     model.fit(X_train, y_train)

# #     train_acc = accuracy_score(
# #         y_train,
# #         model.predict(X_train)
# #     )

# #     val_acc = accuracy_score(
# #         y_val,
# #         model.predict(X_val)
# #     )

# #     print(
# #         "min_samples_split:", split,
# #         "| Train:", train_acc,
# #         "| Val:", val_acc,
# #         "| Depth:", model.get_depth(),
# #         "| Leaves:", model.get_n_leaves()
# #     )



# for leaf in [1, 2, 5, 10, 20, 50, 100, 150]:

#     model = DecisionTreeClassifier(
#         criterion="gini",
#         min_samples_leaf=leaf,
#         random_state=10
#     )

#     model.fit(X_train, y_train)

#     train_acc = accuracy_score(
#         y_train,
#         model.predict(X_train)
#     )

#     val_acc = accuracy_score(
#         y_val,
#         model.predict(X_val)
#     )

#     print(
#         "min_samples_leaf:", leaf,
#         "| Train:", train_acc,
#         "| Val:", val_acc,
#         "| Depth:", model.get_depth(),
#         "| Leaves:", model.get_n_leaves()
#     )



import numpy as np

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)

    probabilities = counts / len(y)

    return -np.sum(probabilities * np.log2(probabilities))


y1 = np.array([1, 1, 1, 1, 1])
y2 = np.array([1, 1, 1, 0, 0])
y3 = np.array([1, 1, 1, 1, 0])
y4 = np.array([1, 1, 0, 0])

print("Pure:", entropy(y1))
print("3/2:", entropy(y2))
print("4/1:", entropy(y3))
print("50/50:", entropy(y4))



import numpy as np


def entropy(y):
    classes, counts = np.unique(y, return_counts=True)

    probabilities = counts / len(y)

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


def information_gain(parent, left, right):

    parent_entropy = entropy(parent)

    left_weight = len(left) / len(parent)
    right_weight = len(right) / len(parent)

    weighted_child_entropy = (
        left_weight * entropy(left)
        + right_weight * entropy(right)
    )

    return parent_entropy - weighted_child_entropy



parent = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0
])

left = np.array([
    1, 1, 1, 1, 0
])

right = np.array([
    1, 0, 0, 0, 0
])

print("Parent entropy:", entropy(parent))
print("Left entropy:", entropy(left))
print("Right entropy:", entropy(right))

print(
    "Information Gain:",
    information_gain(parent, left, right)
)