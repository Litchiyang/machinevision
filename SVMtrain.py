import json
from statistics import mean
import matplotlib.pyplot as plt
with open('output.txt') as fi:
    lst = json.load(fi)

length = len(lst)

#data exploratory 
#area 
areaDict = {}
areaDictMean = {}

for i in range(0,length):
    appleSlice = lst[i]
    #print(appleSlice)
    area = appleSlice[4]
    time = appleSlice[14]
    value = areaDict.get(time)
    if value is None:
        areaDict[time] = []
    areaDict[time].append(area)
#print(areaDict.keys())

for j in range(0,len(list(areaDict.keys()))):
    key = list(areaDict.keys())[j]
    m = mean(areaDict[key])
    areaDictMean[key] = m

areaAve = {}
for k in range(0,len(list(areaDictMean.keys()))):
    key = list(areaDictMean.keys())[k]
    if key<=1:
        if areaAve.get(1) is None:
            areaAve[1] = []
        areaAve[1].append(areaDictMean[key])
    elif key<=2:
        if areaAve.get(2) is None:
            areaAve[2] = []
        areaAve[2].append(areaDictMean[key])
    elif key<=3:
        if areaAve.get(3) is None:
            areaAve[3] = []
        areaAve[3].append(areaDictMean[key])
    elif key<=4:
        if areaAve.get(4) is None:
            areaAve[4] = []
        areaAve[4].append(areaDictMean[key])
    elif key<=5:
        if areaAve.get(5) is None:
            areaAve[5] = []
        areaAve[5].append(areaDictMean[key])
    else:
        if areaAve.get(6) is None:
            areaAve[6] = []
        areaAve[6].append(areaDictMean[key])
print(areaAve)
for key in areaAve.keys():
    length = len(areaAve[key])
    areaAve[key] = sum(areaAve[key])/length
#print(areaDictMean)
plt.bar(areaAve.keys(), areaAve.values(),align='center')
plt.xlabel('Duration Time')
plt.ylabel('Mean of Area')
#plt.xticks(range(len(areaDictMean),areaDictMean.keys()))
plt.show()

#R,G,B show
mean_R = []
mean_G = []
mean_B = []

RDict = {}
RDictMean = {}


for a in range(0,len(lst)):
    appleSlice = lst[a]
    #print(appleSlice)
    meanR = appleSlice[7]
    time = appleSlice[14]
    value = RDict.get(time)
    if value is None:
        RDict[time] = []
    RDict[time].append(meanR)


for j in range(0,len(list(RDict.keys()))):
    key = list(RDict.keys())[j]
    m = mean(RDict[key])
    RDictMean[key] = m

RAve = {}
for k in range(0,len(list(RDictMean.keys()))):
    key = list(RDictMean.keys())[k]
    if key<=1:
        if RAve.get(1) is None:
            RAve[1] = []
        RAve[1].append(RDictMean[key])
    elif key<=2:
        if RAve.get(2) is None:
            RAve[2] = []
        RAve[2].append(RDictMean[key])
    elif key<=3:
        if RAve.get(3) is None:
            RAve[3] = []
        RAve[3].append(RDictMean[key])
    elif key<=4:
        if RAve.get(4) is None:
            RAve[4] = []
        RAve[4].append(RDictMean[key])
    elif key<=5:
        if RAve.get(5) is None:
            RAve[5] = []
        RAve[5].append(RDictMean[key])
    else:
        if RAve.get(6) is None:
            RAve[6] = []
        RAve[6].append(RDictMean[key])
#print(RAve)
for key in RAve.keys():
    length = len(RAve[key])
    RAve[key] = sum(RAve[key])/length
#print(areaDictMean)
plt.bar(RAve.keys(), RAve.values(),align='center')
plt.xlabel('Duration Time')
plt.ylabel('Mean of Red')
#plt.xticks(range(len(areaDictMean),areaDictMean.keys()))
plt.show()


GDict = {}
GDictMean = {}


for a in range(0,len(lst)):
    appleSlice = lst[a]
    #print(appleSlice)
    meanG = appleSlice[8]
    time = appleSlice[14]
    value = GDict.get(time)
    if value is None:
        GDict[time] = []
    GDict[time].append(meanG) 


for j in range(0,len(list(GDict.keys()))):
    key = list(GDict.keys())[j]
    m = mean(GDict[key])
    GDictMean[key] = m

GAve = {}
for k in range(0,len(list(GDictMean.keys()))):
    key = list(GDictMean.keys())[k]
    if key<=1:
        if GAve.get(1) is None:
            GAve[1] = []
        GAve[1].append(GDictMean[key])
    elif key<=2:
        if GAve.get(2) is None:
            GAve[2] = []
        GAve[2].append(GDictMean[key])
    elif key<=3:
        if GAve.get(3) is None:
            GAve[3] = []
        GAve[3].append(GDictMean[key])
    elif key<=4:
        if GAve.get(4) is None:
            GAve[4] = []
        GAve[4].append(GDictMean[key])
    elif key<=5:
        if GAve.get(5) is None:
            GAve[5] = []
        GAve[5].append(GDictMean[key])
    else:
        if GAve.get(6) is None:
            GAve[6] = []
        GAve[6].append(GDictMean[key])
#print(RAve)
for key in GAve.keys():
    length = len(GAve[key])
    GAve[key] = sum(GAve[key])/length
#print(areaDictMean)
plt.bar(GAve.keys(), GAve.values(),align='center')
plt.xlabel('Duration Time')
plt.ylabel('Mean of Green')
#plt.xticks(range(len(areaDictMean),areaDictMean.keys()))
plt.show()

BDict = {}
BDictMean = {}


for a in range(0,len(lst)):
    appleSlice = lst[a]
    #print(appleSlice)
    meanB = appleSlice[9]
    time = appleSlice[14]
    value = BDict.get(time)
    if value is None:
        BDict[time] = []
    BDict[time].append(meanB)


for j in range(0,len(list(BDict.keys()))):
    key = list(BDict.keys())[j]
    m = mean(BDict[key])
    BDictMean[key] = m

BAve = {}
for k in range(0,len(list(BDictMean.keys()))):
    key = list(BDictMean.keys())[k]
    if key<=1:
        if BAve.get(1) is None:
            BAve[1] = []
        BAve[1].append(BDictMean[key])
    elif key<=2:
        if BAve.get(2) is None:
            BAve[2] = []
        BAve[2].append(BDictMean[key])
    elif key<=3:
        if BAve.get(3) is None:
            BAve[3] = []
        BAve[3].append(BDictMean[key])
    elif key<=4:
        if BAve.get(4) is None:
            BAve[4] = []
        BAve[4].append(BDictMean[key])
    elif key<=5:
        if BAve.get(5) is None:
            BAve[5] = []
        BAve[5].append(BDictMean[key])
    else:
        if BAve.get(6) is None:
            BAve[6] = []
        BAve[6].append(BDictMean[key]) 
#print(RAve)
for key in BAve.keys():
    length = len(BAve[key])
    BAve[key] = sum(BAve[key])/length
#print(areaDictMean)
plt.bar(BAve.keys(), BAve.values(),align='center')
plt.xlabel('Duration Time')
plt.ylabel('Mean of Blue')
#plt.xticks(range(len(areaDictMean),areaDictMean.keys()))
plt.show()

#Random Forest Classifier

import pandas as pd 
import collections
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
from sklearn.cross_validation import train_test_split 
from sklearn.linear_model import LinearRegression   
from scipy.stats import probplot
from sklearn.cross_validation import cross_val_score
from tkinter import font
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
from sklearn.metrics import mean_squared_error


X = []
Y = []
for fea in range(0,len(lst)-1):
    apple = lst[fea]
    feature = []
    for f in range(0,14):
        if f==13:
          li = apple[13]
          for lbp in range(0,len(li)-1):
              feature.append(li[lbp])
        else:
            feature.append(apple[f])
    X.append(feature)
    Y.append(apple[14])

for y in range(0,len(Y)):
    ti = Y[y]
    if ti<=1:
        Y[y] = 1
    elif ti<=2:
        Y[y] = 2
    elif ti<=3:
        Y[y] = 3
    elif ti<=4:
        Y[y] = 4
    elif ti<=5:
        Y[y] = 5
    else:
        Y[y] = 6
    
#print(Y)
#X = train[train.columns.difference(['y','profit','review_count','stars'])]
#X = train[train.columns.difference(['roundStars','stars','review_count'])]
#Y = train['y']
X_train,X_test, y_train, y_test = train_test_split(X, Y, random_state=1) 


# model1 = sm.OLS(y_train, X_train).fit()
# predictions = model1.predict(X_test)
# #print(model1.summary())
# prediction_error = y_test - predictions
# print(model1.rsquared)
# # print(type(prediction_error))

#linreg = LinearRegression()  
#model=linreg.fit(X_train, y_train)  
#y_pred = linreg.predict(X_test) 
#mse = mean_squared_error(y_test, y_pred)
#print("mean squared error is: ",mse)
#scores = cross_val_score(model, X_test, y_test, cv=10)
#print("R squared value after 10-fold: ", scores.mean(), scores)



def shapeTest(X_train1,X_test):
    X_test = X_test[X_train1.columns]
    return X_test

clf = RandomForestClassifier(n_estimators=500,random_state=0,n_jobs=-1,oob_score=True)
clf.fit(X_train,y_train)

features = []
#start feature importance visualization
features=['entropy','contrast','energy','correlation','area','circularity','cornerpoints','mean_R','mean_G','mean_B','SD_R','SD_G','SD_B']
for lb in range(0,59):
    string = 'lbp'
    string += str(lb)
    features.append(string)

importances=clf.feature_importances_
indices1 = np.argsort(importances)[::-1][:20]
# indices1=indices[:20]

#plt.figure(1)
#plt.title('Feature Importances')
#plt.barh(range(len(indices1)), importances[indices1], color='b', align='center')
#plt.yticks(range(len(indices1)), features[indices1])
#plt.xlabel('Relative Importance')
#plt.show()

# importances = clf.feature_importances_
headers = ["name", "score"]
values = sorted(zip(features, clf.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))

sfm = SelectFromModel(clf,threshold=0.013)
sfm.fit(X_train,y_train)
print("The number of selected features is: " , len(sfm.get_support(indices=True)))

#X_test2 = shapeTest(X_train,X_test)
X_important_train= sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

clf_important = RandomForestClassifier(n_estimators=500, random_state=1, n_jobs=-1)
clf_important.fit(X_important_train, y_train)

y_important_pred = clf_important.predict(X_important_test)
y_important_pred2 = clf_important.predict(X_important_train)
print("Accuracy is: ", accuracy_score(y_test, y_important_pred))

cm=ConfusionMatrix(y_test,y_important_pred)
cm.plot()
cm.print_stats()
plt.show()
# plt.show()


# y_important_pred2 = clf_important.predict(X_important_train)
# print("Accuracy is: ", accuracy_score(y_train, y_important_pred2))



#print(model.score(X_test,y_test))
# zip(model.feature_cols, model.linreg.coef_)  


#y_pred = linreg.predict(X_test)  
#plt.scatter(y_test, y_pred)
#plt.xlabel('actual score')
#plt.ylabel('predictive score')
#plt.title('model performance')
#plt.show()





