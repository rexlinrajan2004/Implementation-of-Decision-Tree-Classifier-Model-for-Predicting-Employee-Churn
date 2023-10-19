# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:REXLIN R 
RegisterNumber:212222220034
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Data head:
![image](https://github.com/rexlinrajan2004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119406566/a75903c7-f185-42e2-853a-276435f4e6f3)

## Information:
![image](https://github.com/rexlinrajan2004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119406566/74603144-2086-4add-8001-5f0db6c08a4b)

## Null set:
![image](https://github.com/rexlinrajan2004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119406566/7b35f090-016e-4aa8-9ab2-d4a1b477f420)

## Value_counts():
![image](https://github.com/rexlinrajan2004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119406566/f94ef29d-fdee-4b48-a40d-4e17e51cdf2e)

## Data head:
![image](https://github.com/rexlinrajan2004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119406566/56767095-20e1-4b9f-b44a-6885b53a7d33)
## x.head():
![image](https://github.com/rexlinrajan2004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119406566/bcdeb18e-03be-4f84-82a8-91cb909e29dc)
## Data Prediction:
![image](https://github.com/rexlinrajan2004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119406566/f6950012-686f-4982-9b56-b66bbf238449)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
