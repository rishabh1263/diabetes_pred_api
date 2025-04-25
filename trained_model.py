import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
import pickle


## Load the pima dataset using url
url  = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)


## Features and target variable

X = data.drop('Outcome', axis = 1)
y = data['Outcome']

##Split into trainin and testing sets

X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.20, random_state=42)


## Train the model

model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

## Test the model 
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
##Save the model using picket

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
