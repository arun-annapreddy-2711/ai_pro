
#traing the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

#------------------
data_fake=pd.read_csv('./Fake.csv')
data_true=pd.read_csv('./True.csv')

#------------
#print(data_fake,data_true)
#===we are add class for the data init fake class=0 and true class=1
data_fake['class']=0
data_true['class']=1
#print(data_fake,data_true)
#----------------
#=== remove last ten rows
"""
the drop is used for remove i row
axis=0 tell along row wise 
inplace =True from actual data
"""
data_fake_manual_testing=data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i],axis=0,inplace=True)
data_true_manual_testing=data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i],axis=0,inplace=True)

#=====
print(data_fake_manual_testing.shape,data_true_manual_testing.shape)
#----------
#merge the data 10r +10r =20 rowa


merge_data=pd.concat([data_true_manual_testing,data_fake_manual_testing],axis=0)
#print(merge_data)

#---
#remove the coloms given them along coloums wise
da=merge_data.drop(['title','date','subject'],axis=1)

#--
#da
da=da.sample(frac=1)#taking all the data but chaning the order

da.reset_index(inplace=True)#assing index for default row number
da=da.drop(["index",],axis=1)#remove index row

#print(da.columns)


def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)                       # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)         # Remove URLs
    text = re.sub(r'<.*?>+', '', text)                        # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)                            # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)                      # Remove words containing numbers
    return text
#============
x=da['text']
y=da['class']

#--------
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
"""
# here the data x is text and y tell the data is true or false 
test_=0.25=> training 75 and test 25git
# 
"""

#convert the text into vector  for machine understanfing ((embedding))
# #use input only 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)

#
"""
True or False
based on the probalility of it 0 <=0.5 or 1>=0.5 


"""

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)
# print(pred_lr)
LR.score(xv_train,y_train)
# print(LR.score(xv_train,y_train))
a=LR.score(xv_train,y_train)
#print(classification_report(y_test,pred_lr))

#----------
#decision treeclassifier based on classification and regression


from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)
predect_d=DT.predict(xv_test)
# print(predect_d)
# print(DT.score(xv_test,y_test))
b=DT.score(xv_test,y_test)
#print(classification_report(y_test,pred_lr))



#
#this model is correct the error that previous input for next input

from sklearn.ensemble import GradientBoostingClassifier
GB=GradientBoostingClassifier(random_state=0)
GB.fit(xv_train,y_train)
preedit_gb=GB.predict(xv_test)
# print(preedit_gb)
# print(GB.score(xv_test,y_test))
c=GB.score(xv_test,y_test)

#random forest take decion based on the deciontree with random data input
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(random_state=0)
RF.fit(xv_train,y_train)

pred_rf=RF.predict(xv_test)
# print(pred_rf)
# print(RF.score(xv_test,y_test))
d=RF.score(xv_test,y_test)

#---------
print("last")
print(a,b,c,d)
#===========+++++++++++

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    return print(f"\n\nLR Prediction: {output_lable(pred_LR[0])} \
\nDT Prediction: {output_lable(pred_DT[0])} \
\nGBC Prediction: {output_lable(pred_GB[0])} \
\nRF Prediction: {output_lable(pred_RF[0])}")

# newa=input()
# manual_testing(newa)
##True events
print("tre+++++++++++++++++++")
l = []

l.append("The Berlin Wall fell in 1989.")
l.append("Neil Armstrong walked on the moon in 1969.")
l.append("The Titanic sank in 1912.")
l.append("The COVID-19 pandemic began in 2019.")
l.append("Barack Obama was the first African American president of the United States.")
l.append("World War II ended in 1945.")
l.append("The first iPhone was released in 2007.")
l.append("The Wright brothers flew the first powered airplane in 1903.")
l.append("Mahatma Gandhi led India’s independence movement.")
l.append("Mount Everest is the highest mountain above sea level.")
for i in l:
    newa=i
    manual_testing(newa)

#__False wvenets
l = []

l.append("The Great Fire of London happened in 1996.")
l.append("Albert Einstein invented the light bulb.")
l.append("Humans currently live on Mars.")
l.append("The Roman Empire fell in 2020.")
l.append("Shakespeare won a Nobel Prize in Literature.")
l.append("Dinosaurs and humans lived at the same time.")
l.append("Napoleon was extremely short.")
l.append("The internet was invented in the 1800s.")
l.append("The moon is made of cheese.")
l.append("Cleopatra was born in Italy.")
for i in l:
    newa=i
    manual_testing(newa)

