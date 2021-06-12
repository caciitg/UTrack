#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tweepy
import pandas as pd
import numpy as np
import webbrowser
import time
from tweepy import OAuthHandler
import json
import csv
import re
import string
import os


# In[ ]:


key = "VEyxpXLGHG9USYhM7spHVKl36"
secret = "FG61nlBuLR7mb6UCPGxHH4UdMqwYNwL6aFhDt9gQJcaChblOkL"
callback_url = "oob"
auth = tweepy.OAuthHandler(key, secret, callback_url)
redirect_url = auth.get_authorization_url()
webbrowser.open(redirect_url)
pin_input = input("Enter Pin Value : ")
auth.get_access_token(pin_input)


# In[ ]:


api = tweepy.API(auth)


# In[ ]:


lonely_list = 'need help OR lonely OR alone OR feeling lonely OR love me OR dead inside OR i want to die OR #Ineedtotalk OR i need OR all alone'
anxiety_list = "I just can’t OR I’m fine OR Overthinking OR I tried OR I'm okay OR Help me OR I'm fine OR I need OR Left out OR Worry OR Nervous"
stress_list = "very hard OR incredibly OR stressed OR sad OR tired OR It's not easy being OR tension OR selfcare OR insomnia OR trauma OR awake"


# In[ ]:


lonely_tweets = pd.DataFrame(columns = ['username', 'acctdesc', 'location', 'usercreatedts', 'tweetcreatedts',
                                        'retweetcount', 'text', 'hashtags'])
anxiety_tweets = pd.DataFrame(columns = ['username', 'acctdesc', 'location', 'usercreatedts', 'tweetcreatedts',
                                        'retweetcount', 'text', 'hashtags'])
stress_tweets = pd.DataFrame(columns = ['username', 'acctdesc', 'location', 'usercreatedts', 'tweetcreatedts',
                                        'retweetcount', 'text', 'hashtags'])


# In[ ]:


def scraptweets(search_words, numTweets, numRuns, db_tweets):
    
    program_start = time.time()
    for i in range(0, numRuns):
        start_run = time.time()
                
        tweets = tweepy.Cursor(api.search_30_day,environment_name='tweets30days',q=search_words).items(numTweets)
                
        tweet_list = [tweet for tweet in tweets if tweet.lang=='en']    
        noTweets = 0
        
        
        for tweet in tweet_list:
            username = tweet.user.screen_name
            acctdesc = tweet.user.description
            location = tweet.user.location
            usercreatedts = tweet.user.created_at
            tweetcreatedts = tweet.created_at
            retweetcount = tweet.retweet_count
            hashtags = tweet.entities['hashtags']
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                #text = tweet.full_text
                if tweet.truncated:
                    text = tweet.extended_tweet['full_text']
                else:
                     text = tweet.text
            
            ith_tweet = [username, acctdesc, location,
                         usercreatedts, tweetcreatedts, retweetcount, text, hashtags]

            db_tweets.loc[len(db_tweets)] = ith_tweet  
            noTweets += 1
        
        
        end_run = time.time()
        duration_run = round((end_run-start_run)/60, 2)
        
        print('no. of tweets scraped for run {} is {}'.format(i + 1, noTweets))
        print('time take for {} run to complete is {} mins'.format(i+1, duration_run))
        
        time.sleep(5) #15 minute sleep time

    
    program_end = time.time()
    print('Scraping has completed!')
    print('Total time taken to scrap is {} minutes.'.format(round(program_end - program_start)/60, 2))


# In[ ]:


numTweets = 2500
numRuns = 1


# In[ ]:


scraptweets(lonely_list, numTweets, numRuns, lonely_tweets)


# In[ ]:


scraptweets(anxiety_list, numTweets, numRuns, anxiety_tweets)


# In[ ]:


scraptweets(stress_list, numTweets, numRuns, stress_tweets)


# In[ ]:


lonely_tweets['text'] = lonely_tweets['text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)


# In[ ]:


anxiety_tweets['text'] = anxiety_tweets['text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)


# In[ ]:


stress_tweets['text'] = stress_tweets['text'].str.replace(r'[^\x00-\x7F]+', '', regex=True)


# In[ ]:


lonely_tweets.to_csv('lonely_tweets.csv')
anxiety_tweets.to_csv('anxiety_tweets.csv')
stress_tweets.to_csv('stress_tweets.csv')


# In[ ]:


normal_list = '-stress OR -lonely OR -anxious OR -alone OR -sad OR -tension OR -help OR -die OR -miss OR -need'


# In[ ]:


normal_tweets = pd.DataFrame(columns = ['username', 'acctdesc', 'location', 'usercreatedts', 'tweetcreatedts',
                                          'retweetcount', 'text', 'hashtags'])


# In[ ]:


def scraprecenttweets(search_words, numTweets, numRuns, db_tweets):
    
    program_start = time.time()
    for i in range(0, numRuns):
        start_run = time.time()
                
        tweets = tweepy.Cursor(api.search,q=search_words,tweet_mode = 'extended',lang='en').items(numTweets)
                
        tweet_list = [tweet for tweet in tweets]    
        noTweets = 0
        
        
        for tweet in tweet_list:
            username = tweet.user.screen_name
            acctdesc = tweet.user.description
            location = tweet.user.location
            usercreatedts = tweet.user.created_at
            tweetcreatedts = tweet.created_at
            retweetcount = tweet.retweet_count
            hashtags = tweet.entities['hashtags']
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                text = tweet.full_text
                #if tweet.truncated:
                 #   text = tweet.extended_tweet['full_text']
                #else:
                 #   text = tweet.text
            
            ith_tweet = [username, acctdesc, location,
                         usercreatedts, tweetcreatedts, retweetcount, text, hashtags]

            db_tweets.loc[len(db_tweets)] = ith_tweet  
            noTweets += 1
        
        
        end_run = time.time()
        duration_run = round((end_run-start_run)/60, 2)
        
        print('no. of tweets scraped for run {} is {}'.format(i + 1, noTweets))
        print('time take for {} run to complete is {} mins'.format(i+1, duration_run))
        
        time.sleep(5) #15 minute sleep time

    
    program_end = time.time()
    print('Scraping has completed!')
    print('Total time taken to scrap is {} minutes.'.format(round(program_end - program_start)/60, 2))


# In[ ]:


numTweets_1 = 2000
numRuns_1 = 1


# In[ ]:


scraprecenttweets(normal_list, numTweets_1, numRuns_1, normal_tweets)


# In[ ]:


normal_tweets.to_csv('normal_tweets.csv')


# In[ ]:


lonely_tweets.to_csv('lonely_tweets_2.csv')
anxiety_tweets.to_csv('anxiety_tweets_2.csv')
stress_tweets.to_csv('stress_tweets_2.csv')


# In[ ]:




