
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk import WordNetLemmatizer
import pickle


# In[ ]:

lemmatizer = WordNetLemmatizer()


# In[ ]:

# Loading Data from json that was created after wrangling


# In[ ]:

#data.reset_index(inplace=True)


# In[ ]:

#data.tail()


# In[ ]:

'''X = data.INPUTS
Y = data.TARGET
part1 = data.iloc[0:120000]
part2 = data.iloc[120001:]'''


# In[ ]:

'''X1 = part1.INPUTS
Y1 = part1.TARGET'''


# In[ ]:

'''X2 = part2.INPUTS
Y2 = part2.TARGET
'''

# In[ ]:

with open('lexicon.txt','rb') as fl:
    lexicon = pickle.load(fl)


# In[ ]:

'''len(lexicon)


# In[ ]:

len(classification1)'''


# In[3]:

def load_data(data):
    data1= pd.read_json(data)
    data1.reset_index(inplace=True)
    return data1.INPUTS, data1.TARGET


# In[2]:


def prepare(X):

    feature_set = []

    i=0

    for line in X:


        sample_words = word_tokenize(line)
        sample_words = [lemmatizer.lemmatize(i) for i in sample_words]

        features = np.zeros(len(lexicon))
        #classification = np.zeros(len(lexicon))
        for word in sample_words:
            if word in lexicon:
                features[lexicon.index(word)]+=1
        feature_set.append(features)
    return feature_set


# In[4]:

def create_labels(data):
    X,Y = load_data(data)
    return prepare(X), prepare(Y)


# In[ ]:



