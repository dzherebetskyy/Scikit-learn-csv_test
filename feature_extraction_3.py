import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

#?? deals with the empy lines
fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)
#takes lines in table and split it into columns 

# gives where and how many times a "text" occurs in the record 
#print count_vect.transform(["I love my iphone!!!"])
#print count_vect.transform(["I love my iphone!!!", "I hate my iphone!!"])

#number of times iphone occurs in the text
print count_vect.transform(["IPhone"])


#prints the first row
#print counts[0]

#prints the twit in the first row  
#print fixed_text[0]

#prints the whole vocabulary
#print count_vect.vocabulary_