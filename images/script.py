import kagglehub
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download latest version
path = kagglehub.dataset_download("jacksondivakarr/car-crash-dataset")

#print("Path to dataset files:", path)

crash_datased = pd.read_csv('/monroe county car crach 2003-2015 - monroe county car crach 2003-2015.csv.csv', encoding='latin-1')

X = crash_datased[['Collision Type','Hour']]

y = crash_datased['Weekend?']
#print(X)
X = X.dropna()
y = y.loc[X.index]
print(X)
print(y)

model = DecisionTreeClassifier()
model.fit(X,y)
predictions = model.predict([ [2,1420.0],[3,1250.0] ])
print(predictions)

""" **What I'm doing?**
I'm allocating in lists the 20% of dataset to know the accuracy of the model that i make.

 **How Works**
I'm creating 2 testing list for X and y e 2 list trained for X and y.
Later i will compare trained list and test list, and we can know the accuracy.

"""

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)

model.fit(X_train,y_train)
predictions = model.predict(X_test) #prediction return a y_prediction_value

accuracy = accuracy_score(y_test,predictions)

print(f'Accuracy of the prediction: {round(accuracy,2)*100}%')

"""**Last Step**

Now I make the model persistent, if I train it once, it doesn't need retraining.

If a person learns a chapter of a book, why should he relearn it?
"""


import kagglehub
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

#joblib.dump(model, 'crashes-predictions.joblib') File with data already trained
model = joblib.load('crashes-predictions.joblib')
predictions = model.predict([ [2,1420.0],[3,1250.0] ])
print(predictions)

