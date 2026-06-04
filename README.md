# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the spam dataset and preprocess the data by selecting message and label columns.
2.Convert text messages into numerical form using TF-IDF Vectorizer.
3. Split the dataset into training and testing data, then train the SVM classifier.
4.Predict the output using the trained model and display the confusion matrix using a heatmap. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Ramsanjay C
RegisterNumber:  212224220077
*/
```
```
import pandas as pd
import chardet
with open('spam.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print(result)

data = pd.read_csv('spam.csv',
                   encoding=result['encoding'])

print(data.head())
print(data.info())
print(data.isnull().sum())


X = data['v2'].values      
y = data['v1'].values      

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy =", accuracy)
```
## Output:
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/bafeb5ce-9805-4825-9301-b258cea68e9e" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
