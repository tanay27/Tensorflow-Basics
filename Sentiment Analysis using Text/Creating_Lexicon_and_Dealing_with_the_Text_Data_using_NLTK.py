
# coding: utf-8

# In[2]:

'''
Lexicon:

Let suppose I have the dictionary as list with items ['fifa','world','cup','going','awesome']
sentence = 'I am watching fifa world cup and it is going awesome'
,
then the sentence can be represented in terms of dictionary and numerics to deal with as follows:

sentence vector representation:
[1,1,1,1,1] as all the dictionary words are present in the sentence
NLTK is a great way to do that as it has the dictionary that can be used 
for the bag of words we are aiming.

tokenize:  sentence--->[I, am, watching, fifa, world, cup, and, it, is, going, awesome]
lemmatize: watching, watch, watches---> basically watch, convert to form where it has
           true meaning in the dictionary

'''
import nltk


# In[3]:

from nltk.tokenize import word_tokenize


# In[4]:

from nltk.stem import WordNetLemmatizer


# In[6]:

import numpy as np
import random
import pickle
from collections import Counter


# In[16]:

#Lines upto how many you want to read.
lemmatizer = WordNetLemmatizer()
Lines = 1000000


# In[17]:

def lexicon(pos, neg):
    lexicon = []
    for file in [pos,neg]:
        with open(file,'r') as f:
            content =f.readlines()
            for line in content[:Lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(n) for n in lexicon]
    count_words = Counter(lexicon) #Counter counts how many times each word has repeated
    
    lexicon_final = []
    for word in count_words:
        if 1000 > count_words[word] > 50:
            lexicon_final.append(word)
            
    return lexicon_final


# In[25]:

def process_new_sample(sample,lexicon, classification ):
    
    feature_set = []
    
    
    with open(sample,'r') as f:
        
        content = f.readlines()
        for line in content[:Lines]:
            
            sample_words = word_tokenize(line.lower())
            sample_words = [lemmatizer.lemmatize(i) for i in sample_words]
            features = np.zeros(len(lexicon))
            for word in sample_words:
                
                if word.lower() in lexicon:
                    idx = lexicon.index(word.lower())
                    features[idx]+=1
            features = list(features)
            feature_set.append([features, classification])
    return feature_set


# In[30]:

def create_labels(pos, neg, test_size = 0.1):
    
    
    Lexicon = lexicon(pos, neg)
    features = []
    features+=process_new_sample('pos.txt', Lexicon, [1,0])
    features+=process_new_sample('neg.txt', Lexicon, [0,1])
    
    random.shuffle(features) #shuffling is important to train and test for numerous reasons
    Test_Size = int(test_size*len(features))
    
    '''features-->
    [[feature, label]]
    
    '''
    features = np.array(features)
    train_x = list(features[:,0][:-Test_Size])
    train_y = list(features[:,1][:-Test_Size])
    
    test_x = list(features[:,0][-Test_Size:])
    test_y = list(features[:,1][-Test_Size:])
    
    return train_x, train_y, test_x, test_y


# In[31]:

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_labels('pos.txt','neg.txt')
    with open('sentiment.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


# ## Processing is done! We have our pickle file.
# ## In the next notebook, we will use this to train our neural network using tensorflow
