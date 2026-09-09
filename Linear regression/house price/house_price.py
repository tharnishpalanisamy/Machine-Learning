import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('./house_price.csv') 

x = df[['area']] 
y = df['price'] 

X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42) 

model = LinearRegression()  
model.fit(X_train , y_train ) 

prediction = model.predict(X_test)  
 
print('Actual results' ) 
print(y_test) 

print('Predicted Results') 
print(prediction)  

mae = mean_absolute_error(y_test , prediction) 

print('Mean absolute Error' , mae)


prediction2 = model.predict(  pd.DataFrame({"area": [12341234]}) ) 
print('pred2 : ' , prediction2)

