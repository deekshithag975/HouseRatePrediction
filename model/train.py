import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

#loading dataset
df=pd.read_csv("data/houserates.csv")

#input data and output
X=df[["area","bed_rooms","age"]]
y=df["price"]

#create model
model = LinearRegression()

#train the model with i/p and o/p features
model.fit(X,y)

#display learned parameters
print("Slope - m(weight) = ",model.coef_)
print("Intercept - c(constant) = ",model.intercept_)

#test prediction
sq_feet = [[1500,3,5]]
pred_price=model.predict(sq_feet)
print("Predicted Price = ",pred_price[0])

#save model
joblib.dump(model,"model/house_price_model.pkl")
print("Model Saved Successfully")