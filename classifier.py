import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(counts, fixed_target)
#fit learns on classifier
#it takes our words, looks in all tweets for occurance of our words and calculates association of these words with positive and negative emotions, and gives most probable results  

print nb.predict(count_vect.transform(["I love my iphone!!!"]))
print nb.predict(count_vect.transform(["I hate my iphone!!!"]))
print nb.predict(count_vect.transform(["I don't my iphone!!!"]))  #makes positive emotion
#print nb.predict(count_vect.transform(["iphone cost too much!!!"]))
