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


#data import
from google.colab import files
uploaded = files.upload()


# In[ ]:


df = pd.read_csv(io.BytesIO(uploaded['final_stress (1).csv']))


# In[ ]:


df = df.drop(columns = ['Unnamed: 0'])


# In[ ]:


module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)


# In[ ]:


#preprocessing
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
train_input = bert_encode(df.cleaned_text.values, tokenizer, max_len=max_len)
train_labels = np.array(df.target.values)


# In[ ]:


test_input = bert_encode(df.cleaned_text[:10].values, tokenizer, max_len=max_len)


# In[ ]:


model = build_model(bert_layer, max_len=max_len)
model.summary()


# In[ ]:


checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
train_history = model.fit(train_input, train_labels, validation_split = 0.25, epochs = 10, callbacks=[checkpoint, earlystopping], verbose = 1,batch_size = 16)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


model.save('/content/drive/MyDrive/UTrack_Models/UTrack_Stress') 

