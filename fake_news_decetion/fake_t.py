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
