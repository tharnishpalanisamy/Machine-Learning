import pandas as pd   
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier , plot_tree , export_text 

df = pd.DataFrame({
    'Age' : [5,8,12,20,25,30]  , 
    'Survived' : [1,1,0,0,0,1]
})


print(df) 
X = df[['Age']] 
y = df['Survived'] 

model  = DecisionTreeClassifier(max_depth=3 , random_state= 42 ) 

model.fit(X , y ) 

# pred = model.predict([[18]]) 

# print(pred)

# plt.figure(figsize=(12,6))

# plot_tree(
#     model , 
#     feature_names=['Age'] ,  
#     class_names=['Died' , 'Survived'] , 
#     filled=True 

# )
# print(model.get_depth())
# print(model.get_n_leaves())

# plt.show() 

tree1 = DecisionTreeClassifier(max_depth=2 , random_state=42) 
tree2 = DecisionTreeClassifier(max_depth=2 , random_state=42)  
tree3 = DecisionTreeClassifier(max_depth=3 , random_state=42)   

tree1.fit(X,y)
tree2.fit(X,y)
tree3.fit(X,y)

# print('Depth 1 : ' , tree1.get_depth() , 'Leafs : ' , tree1.get_n_leaves())
# print('Depth 2 : ' , tree2.get_depth() , 'Leafs : ' , tree2.get_n_leaves())
# print('Depth 3 : ' , tree3.get_depth() , 'Leafs : ' , tree3.get_n_leaves()) 

# for age in [6, 10, 15, 22, 28]:
#     sample = pd.DataFrame({"Age": [age]})
#     print(
#         age,
#         tree1.predict(sample),
#         tree2.predict(sample),
#         tree3.predict(sample)
#     )



print(export_text(tree1, feature_names=["Age"]))
