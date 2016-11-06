# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 09:33:08 2016

@author: Ab6ge1
"""

import pandas as pd
import numpy as np
import sklearn 

#print "hello world"

#download data from
#bit.ly/crowdflower-data
#https://github.com/lukas/scikit-class

# LOAD DATA
df = pd.read_csv('1377884607_tweet_product_company.csv')
#or full path
#df = pd.read_csv('/media/c/!Danylo/Datascience/Machinelerning_Galvanize/1377884607_tweet_product_company.csv')
#df - stands for data frame
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text= df['tweet_text']

#print target[0:5]
#print text[0:5]

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
count_vect.fit(text)

#get number of tymes the word appears in the table
print count_vect.vocabulary_.get(u'3g')
print count_vect.vocabulary_.get(u'iphone')