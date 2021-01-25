#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()
os.chdir ('/Users/Aishu/Desktop')
os.getcwd()


# In[2]:


pip install --user -U nltk


# In[3]:


import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')


# In[1]:


import re
import seaborn as sbn
import nltk
import tqdm as tqdm
import sqlite3
import pandas as pd
import numpy as np

import string
import nltk
from nltk.corpus import stopwords
stop = stopwords.words("english")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from math import floor,ceil
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

from sklearn.svm import LinearSVC

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding


from gensim import summarization
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[2]:


review_data = pd.read_json('/Users/Aishu/Desktop/reviews_Beauty_5.json',lines=True)


# In[3]:


review_data.head()


# In[4]:


review_data.columns


# In[5]:


review_data = review_data.drop(["helpful","reviewTime","reviewerName","unixReviewTime"],axis = 1)


# In[6]:


review_data.head()


# In[7]:


sbn.countplot(review_data['overall'])


# In[8]:


def data_clean( rev, remove_stopwords=True): 
    

    new_text = re.sub("[^a-zA-Z]"," ", rev)
   
    words = new_text.lower().split()
    
    if remove_stopwords:
        sts = set(stopwords.words("english"))
        words = [w for w in words if not w in sts]
    ary=[]
    eng_stemmer = english_stemmer 
    for word in words:
        ary.append(eng_stemmer.stem(word))

    
    return(ary)


# In[9]:


clean_reviewData = []
for rev in review_data['reviewText']:
    clean_reviewData.append( " ".join(data_clean(rev)))
    
clean_summaryData = []
for rev in review_data['summary']:
    clean_summaryData.append( " ".join(data_clean(rev)))


# In[10]:


Most_used_Words_Review =pd.Series(' '.join(clean_reviewData).lower().split()).value_counts()[:25]
print (Most_used_Words_Review)


# In[11]:


Most_used_Words_Summary = pd.Series(' '.join(clean_summaryData).lower().split()).value_counts()[:20]
print (Most_used_Words_Summary)


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
text_vectorizer = TfidfVectorizer(min_df=4, max_features = 1000)
test_vecor = text_vectorizer.fit_transform(clean_reviewData)
tfidf_vector = dict(zip(text_vectorizer.get_feature_names(), text_vectorizer.idf_))


# In[13]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
sample_review = review_data.reviewText[:10]
for test in sample_review:
    test
    ss = analyser.polarity_scores(test)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]))
    print(test)


# In[14]:


from sklearn.cluster import MiniBatchKMeans

clusters = 20
kmeans_model = MiniBatchKMeans(n_clusters=clusters, init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
kmodel = kmeans_model.fit(test_vecor)
kmodel_clusters = kmodel.predict(test_vecor)
kmodel_distances = kmodel.transform(test_vecor)
centroids = kmodel.cluster_centers_.argsort()[:, ::-1]
values = text_vectorizer.get_feature_names()
for i in range(clusters):
    print("Cluster %d:" % i)
    for j in centroids[i, :5]:
        print(' %s' % values[j])
    print()


# In[15]:


test_reviewText = review_data.reviewText
test_Ratings = review_data.overall
text_vectorizer = TfidfVectorizer(max_df=.8)
text_vectorizer.fit(test_reviewText)
def rate(r):
    ary2 = []
    for rating in r:
        tv = [0,0,0,0,0]
        tv[rating-1] = 1
        ary2.append(tv)
    return np.array(ary2)


# In[16]:


test_reviewText =test_reviewText[:2000]
test_reviewText


# In[17]:


test_Ratings = test_Ratings[:2000]
test_Ratings


# In[18]:


X = text_vectorizer.transform(test_reviewText).toarray()
y = rate(test_Ratings.values)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

model = Sequential()
model.add(Dense(128,input_dim=X_train.shape[1]))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=10,batch_size=32,verbose=1)
model.evaluate(X_test,y_test)[1]


# In[ ]:




