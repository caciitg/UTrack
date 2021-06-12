#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import re
import string
import random
from nltk.tokenize import WordPunctTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus im pplport stopwords


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[ ]:


from google.colab import files 
uploaded = files.upload()


# In[ ]:


def extract_csv():
    my_filtered_csv = pd.read_csv('clown_1.csv', usecols=['tweet'])
    return my_filtered_csv

def tokenize_tweets(clown_1) :
    tweets = clown_1.tweet.tolist()
    tokenizer = WordPunctTokenizer() 
    cleaned = []
    for i in range(0, len(tweets)):
        text = tweets[i]
        text = re.sub('^https?://.*[rn]*','', text)
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        text = re.sub("(@[A-Za-z0-9_]+)","", text)
        text = re.sub("([^\w\s])", "", text)
        text = re.sub("^RT", "", text)
        text = tokenizer.tokenize(text)
        element = [text]
        cleaned.append(element)
    return cleaned

def lemmatize_sentence(tweet_tokens, stop_words = ()):
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('V'):
            pos = 'v'
        else:
            pos = 'a'
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def create_lemmatized_sent(words):
    cleaned = []
    stop_words = stopwords.words('english')
    for i in range(0, len(words)):
        sent = lemmatize_sentence(words[i][0], stop_words)
        if len(sent) >= 0:
            element = [sent]
            cleaned.append(element)
    return cleaned

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def write_sent(sent):
    cleaned = []
    for i in sent:
        s = ""
        for j in i[0]:
            j = str(j)
            j = j + " "
            s = s + j
        s = remove_emoji(s)
        element = [s]
        cleaned.append(element)
    df = pd.DataFrame(cleaned)
    # print(df.iloc[1])
    #df.to_csv('cleaned_clown_1.csv', index=False)
    df1 = pd.read_csv('clown_1.csv')
    df1 = df1['time']
    big = pd.concat([df, df1], axis = 1)
    big.to_csv('cleaned_clown_1.csv', index=False)

clown_1 = extract_csv()
words = tokenize_tweets(clown_1)
sent = create_lemmatized_sent(words)
write_sent(sent)


# In[ ]:


df = pd.read_csv('cleaned_clown_1.csv')
df = df.rename(columns={"0": "clean_tweet"})
df


# In[ ]:


df.to_csv('cleaned_clown_1.csv', index=True) 
files.download('cleaned_clown_1.csv')


# In[ ]:




