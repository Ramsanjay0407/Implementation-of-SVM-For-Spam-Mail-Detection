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
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = "spam.csv"   

with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print("Detected Encoding:", result)

data = pd.read_csv(file_path, encoding=result['encoding'])

print(data.head())
print(data.info())
print(data.isnull().sum())

plt.figure(figsize=(5,4))
sns.countplot(x=data['v1'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

data['msg_length'] = data['v2'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(data=data, x='msg_length', hue='v1', bins=50, kde=True)
plt.title("Message Length Distribution (Spam vs Ham)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

x = data['v2'].values     
y = data['v1'].values     
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

svc = SVC(kernel='linear')
svc.fit(x_train_vec, y_train)

y_pred = svc.predict(x_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix – SVM Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
```
## Output:

<img width="1920" height="1200" alt="Screenshot (60)" src="https://github.com/user-attachments/assets/0d0316ee-b8b1-4376-a30c-c6aa16e63a1f" />
<img width="1920" height="1200" alt="Screenshot (61)" src="https://github.com/user-attachments/assets/e84b19ed-3d9f-4f68-b5c3-b3737914c295" />
<img width="1920" height="1200" alt="Screenshot (62)" src="https://github.com/user-attachments/assets/18cbffaf-6a1e-4f7f-aec3-34401d1a8be6" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
