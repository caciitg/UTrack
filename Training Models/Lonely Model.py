#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow_hub as hub
import io
from keras.layers import Input, Dropout, Dense, Activation
from keras.utils import to_categorical
import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO)


# In[ ]:


pip install -q tf-models-official==2.3.0


# In[ ]:


from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.tokenization as tokenization

from sklearn.preprocessing import LabelEncoder


# In[ ]:


np.random.seed(321)


# In[ ]:


lonely_df = pd.read_csv("big_lonely.csv")
normal_df = pd.read_csv("df_normal_edited.csv")

lonely_df.drop('Unnamed: 0', inplace=True, axis=1)
normal_df.drop('Unnamed: 0', inplace=True, axis=1)

lonely_df['target'] = '1'
normal_df['target'] = '0'

lonely_df.rename(columns={'0':'cleaned_text'}, inplace=True)

combo_df = pd.concat([lonely_df, normal_df])
combo_df.reset_index(drop=True, inplace=True)

combo_df = combo_df.sample(frac=1, random_state=321).reset_index(drop=True)

train_data = combo_df.sample(frac = 0.8, random_state=321)
test_data = combo_df.drop(train_data.index) 

y_df = combo_df['target']
x_df = combo_df.drop('target', axis=1)

x_df.reset_index(drop=True, inplace=True)
y_df.reset_index(drop=True, inplace=True)


# In[ ]:


labelencoder = LabelEncoder()
combo_df= combo_df.copy()
combo_df.target = labelencoder.fit_transform(combo_df.target)


# In[ ]:


module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)


# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


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


# In[ ]:


def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(1, activation=None)(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


max_len = 150
train_input = bert_encode(combo_df.cleaned_text.values, tokenizer, max_len=max_len)
train_labels = np.array(combo_df.target.values)


# In[ ]:


model = build_model(bert_layer, max_len=max_len)
model.summary()


# In[ ]:


checkpoint = tf.keras.callbacks.ModelCheckpoint('lonely.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
train_history = model.fit(train_input, train_labels, validation_split = 0.25, epochs = 10, callbacks=[checkpoint, earlystopping], verbose = 1,batch_size = 32)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


model.save('/content/drive/MyDrive/UTrack_Models/UTrack_Lonely') 

