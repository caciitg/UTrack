#!/usr/bin/env python
# coding: utf-8

# ## Streamlit Web Deployment
# Run this file as .ipynb notebook, cells division is specified

# In[ ]:

!pip install -q tf-models-official==2.3.0
!pip install streamlit
!pip install pyngrok


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


%%writefile utilss.py
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
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tweepy
consumerKey = "VEyxpXLGHG9USYhM7spHVKl36"
consumerSecret = "FG61nlBuLR7mb6UCPGxHH4UdMqwYNwL6aFhDt9gQJcaChblOkL"
accessToken = "1142865475459846145-5VQ9CRY7iRlneurWdNzwHmT4Y9k6L1"
accessTokenSecret = "iAkL3XWrsBdBQWn2eH8ifqnjoWkvBF5EHvJ1SsH6EcfLB"

authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
authenticate.set_access_token(accessToken, accessTokenSecret) 
api = tweepy.API(authenticate, wait_on_rate_limit = True)
from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.tokenization as tokenization
import PIL
import pandas as pd
import numpy as np
import io
import tensorflow_hub as hub

from keras.layers import Input, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

model_lonely = keras.models.load_model('/content/drive/MyDrive/Utrack_Models/Utrack_Lonely')

def Show_Recent_Tweets(raw_text):
  posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")   
  def get_tweets():
    column_names = ['tweet', 'time']
    user = pd.DataFrame(columns =column_names)
    
    tweet_time = []
    tweet_text = []
    for info in posts[:100]:
      tweet_time.append(info.created_at)
      tweet_text.append(info.full_text)
    
    user['time'] = tweet_time
    user['tweet'] = tweet_text
    
    return user
 
  recent_tweets=get_tweets()        
  return recent_tweets
 
def tokenize_tweets(clown) :
    tweets = clown.tweet.tolist()
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

def write_sent(clown, sent):
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
    df = pd.DataFrame(cleaned,columns = ['text'])
    df1 = clown
    df1 = df1['time']
    big = pd.concat([df, df1], axis = 1)
    return big
  
def import_and_predict(df, model):
  max_len = 150
  
  test_input = bert_encode(df["text"].values, tokenizer, max_len=max_len)
  prediction = model.predict(test_input)

  return prediction


def output_dataframe(df, prediction):
  df2 = pd.concat([df, prediction], axis = 1)
  return df2

def visualisation(n, file):
  #constructing data
  df = file
  jscolumn = df.predictions

  df['final'] = jscolumn
  df['perc'] = 100*(df.final)
  new_df = df.drop(columns = ['predictions','final'])
  plt.figure(figsize=(40,15))
  n = int(input())
  temp_df = new_df[:n]
  final_df  = temp_df.iloc[::-1]
  sns.lineplot(x='time', y='perc', data=final_df, linewidth=7, color = 'red')
  plt.title("Mental State vs Date", fontsize= 40,fontweight='bold')
  sns.set_style('white')

  plt.xlabel('Month',fontsize=30,fontweight='bold')
  plt.xticks(fontsize=20,rotation=90)
  plt.ylabel('Percentage',fontsize=30,fontweight='bold')
  plt.yticks(fontsize=25)
  plt.grid(axis='y', alpha=0.5)

  st.pyplot()

  monthdict = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8,
                      "September":9, "October":10, "November":11, "December":12}
  values= []
  for month in df.months.unique():
    dftempo = df[pd.to_datetime(df['time']).dt.month == monthdict[month]]
    values.append(jsmean(dftempo.final))
      
  plt.figure(figsize=(15,10))
  x= df.months.unique()
  height = 100*np.array(values)
  plt.bar(x, height, width=0.5, bottom=None, align='center', color=['#78C850',  # Grass
                      '#f20a53',  # Fire
                      '#6890F0',  # Water
                      '#A8B820',  # Bug
                      '#A8A878',  # Normal
                      '#A040A0',  # Poison
                      '#F8D030',  # Electric
                      '#E0C068',  # Ground
                      '#EE99AC',  # Fairy
                      '#C03028',  # Fighting
                      '#6cf5d3',                                                               
                      '#561191'
                    ])


  sns.set_style('white')


  plt.xlabel('Month',fontsize=15,fontweight='bold')
  plt.xticks(fontsize=15,rotation=45)
  plt.ylabel('Percentage',fontsize=15,fontweight='bold')
  plt.yticks(fontsize=15)
  plt.grid(axis='y', alpha=0.5)
  plt.title('Average Percentage across months', fontsize=20)

  st.pyplot()

  df['months'] = df['time'].dt.month_name()
  plt.figure(figsize=(10,5))
  sns.set_style('white')
  sns.swarmplot(x='months', y='perc', data=df.iloc[::-1])
  #plt.xticks(rotation=90);
  plt.xlabel('Month',fontsize=15,fontweight='bold')
  plt.xticks(fontsize=15,rotation=0)
  plt.ylabel('Percentage',fontsize=15,fontweight='bold')
  plt.yticks(fontsize=15)
  plt.title('Percentage across months', fontsize=20)
  plt.grid(axis='y', alpha=0.5)

  st.pyplot()

  plt.figure(figsize=(10,6))
 
  sns.violinplot(x='months',
                y='perc', 
                data=df.iloc[::-1], 
                inner=None)
  
  sns.swarmplot(x='months', 
                y='perc', 
                data=df.iloc[::-1], 
                color='k',
                alpha=1) 
  plt.grid(axis='y', alpha=0.5)
  plt.xlabel('Month',fontsize=15,fontweight='bold')
  plt.xticks(fontsize=15,rotation=0)
  plt.ylabel('Percentage',fontsize=15,fontweight='bold')
  plt.yticks(fontsize=15)
  plt.title('Percentage across months', fontsize=20)

  st.pyplot()

  def make_pie(sizes, text, colors):
    import matplotlib.pyplot as plt
    import numpy as np
    sizes = [100-100*jsmean(df['final']), 100*jsmean(df['final'])]
    text = round(jsmean(df['final'])*100,2)
    col = [[i/255. for i in c] for c in colors]

    fig, ax = plt.subplots()
    ax.axis('equal')
    width = 0.30
    kwargs = dict(colors=col, startangle=90)
    outside, _ = ax.pie(sizes, radius=1, pctdistance=1-width/2,**kwargs)
    plt.setp( outside, width=width, edgecolor='white')

    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)
    plt.show()

  c2 = (226,33,0)
  c1 = (40,133,4)

  make_pie([257,90], round(df['perc'].mean(), 2),[c1,c2])
  
  st.pyplot()

def probability_out(x):
  n=len(x)
  for i in range(n):
    if(x.iloc[i,0]<0): 
      x.iloc[i,0]=0
    if(x.iloc[i,0]>=0 and x.iloc[i,0]<=1):
      x.iloc[i,0]=np.sin(x.iloc[i,0])
    if(x.iloc[i,0]>1):
      x.iloc[i,0]=(np.log(x.iloc[i,0])+(np.pi)*(np.pi)*(np.sin(1)))/((np.pi)**2)
    if(x.iloc[i,0]>1):
      x.iloc[i,0]=1
  return x

def tweets_conclusion(df):
  #compute weights
  def weight(x):
    return (np.exp(x)-1)/(np.exp(1)-1)

  def jsmean(arr):
    num = 0
    den = 0
    for i in arr:
        den = den + weight(i)
        num = num + i*weight(i)
    return (num/den)[0]

  new_df = df.values
  return jsmean(new_df)

def combine_all(user_name):
  #preprocessing input data
  raw_text = user_name
  recent_tweets=Show_Recent_Tweets(raw_text)
  words = tokenize_tweets(recent_tweets)
  sent = create_lemmatized_sent(words)
  df = write_sent(recent_tweets, sent)
  #loading models
  
  us = f"Setting up models for analysing the profile of **{api.get_user(screen_name=raw_text).name}**"
  st.markdown(us)

  st.text("Loading the model")
  model_lonely = keras.models.load_model('/content/drive/MyDrive/Utrack_Models/Utrack_Lonely')
  model_stress = keras.models.load_model('/content/drive/MyDrive/Utrack_Models/Utrack_Stress')
  model_anxiety = keras.models.load_model('/content/drive/MyDrive/Utrack_Models/Utrack_Anxiety')
  
  intro = f"Twitter Bio of the user =>  **{api.get_user(screen_name=raw_text).description}**"
  st.markdown(intro)

  bio = f"User lives in **{api.get_user(screen_name=raw_text).location}**"
  st.markdown(bio)

  fol = f"Number of Followers of the user => **{api.get_user(screen_name=raw_text).followers_count}**"
  st.markdown(fol)

  st.text("Hold Up!! Working on Predictions...")

  prediction_lonely = import_and_predict(df, model_lonely)
  prediction_stress = import_and_predict(df, model_stress)
  prediction_anxiety = import_and_predict(df, model_anxiety)

  st.text("Predictions Done")
  
  col1, col2, col3 = st.beta_columns(3)

  prediction_lonely = pd.DataFrame(prediction_lonely, columns = ['Loneliness'])
  prediction_stress = pd.DataFrame(prediction_stress, columns = ['Stress'])
  prediction_anxiety = pd.DataFrame(prediction_anxiety, columns = ['Anxiety'])
  
  prediction_lonely = probability_out(prediction_lonely)
  prediction_stress = probability_out(prediction_stress)
  prediction_anxiety = probability_out(prediction_anxiety)
  
  df_total = output_dataframe(df,prediction_lonely)
  df_total = output_dataframe(df_total,prediction_stress)
  df_total = output_dataframe(df_total,prediction_anxiety)

  st.write(df_total)
  df_total = df_total.rename(columns={'time':'index'}).set_index('index')
  
  with col1:
    st.text("LONELINESS LEVELS")
    st.success(tweets_conclusion(prediction_lonely))
    st.line_chart(data=df_total['Loneliness'])

  with col2:
    st.text("STRESS LEVELS")
    st.success(tweets_conclusion(prediction_stress))
    st.line_chart(data=df_total['Stress'])
  
  with col3:
    st.text("ANXIETY LEVELS")
    st.success(tweets_conclusion(prediction_anxiety))
    st.line_chart(data=df_total['Anxiety'])


# In[ ]:


%%writefile app.py
from utilss import combine_all
import tensorflow as tf
import streamlit as st
from tensorflow import keras
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
     page_title="UTrack",
     layout="wide"
)
st.title("UTrack")

st.subheader('*Analysing Twitter Users on Tweet-to-Tweet basis to track levels of Loneliness, Stress & Anxiety*')

raw_text = st.text_input("Enter the exact twitter handle of the Personality (without @)")
st.text(raw_text)
if raw_text == '':
  st.text('Enter userID')
else:
  combine_all(raw_text)

  


# ## Running  localhost server for colab from ngrok

# In[ ]:


!ngrok authtoken 1pqPDOU30ORUzHtrlCA5DX7odxX_4N3in7gRue2ctUDTBYPun


# In[ ]:


!nohup streamlit run app.py --server.port 80 &


# In[ ]:


from pyngrok import ngrok

url = ngrok.connect(port=80)
url


# In[ ]:


!cat /content/nohup.out


# In[ ]:


# Uncomment this only to kill the terminals.
## ! killall ngrok


# In[ ]:




