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

# first 6000 counts we TRAIN our model
nb.fit(counts[0:6000], fixed_target[0:6000])

#use other records  to TEST your model
predictions = nb.predict(counts[6000:9092])
print sum(predictions == fixed_target[6000:9092])
#2053/3092=66% - accuracy, because we have 2053 predictions out of 3092=9092-6000
