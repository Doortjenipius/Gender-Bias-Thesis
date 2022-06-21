#!/usr/bin/env python
# coding: utf-8

# ## Using Natural Language Processing to map gender bias in news articles about terrorism
# 
# ### University of Amsterdam
# #### MA Thesis New Media and Digital Culture 
# ##### Doortje Nipius
# 
# 
# *The original dataset is property of the Simon Fraser University and is protected, handled and used according to the signed agreement.*

# **Data preprocessing**

# In[1]:


#Import libraries 
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


# ### Dataset part 1

# In[2]:


df_1 = pd.read_csv('GGT_deel1.csv')


# In[3]:


#Filter some columns
df_1_filter = df_1[['body', 'outlet' , 'publishedAt', 'people', 'authorsFemaleCount', 'authorsMaleCount','authorsUnknownCount', 'peopleCount', 'peopleFemaleCount', 'peopleMaleCount',
       'peopleUnknownCount','sourcesFemale', 'sourcesFemaleCount', 'sourcesMale',
       'sourcesMaleCount', 'sourcesUnknown', 'sourcesUnknownCount' , 'voicesFemale',
       'voicesMale']]


# In[4]:


# Remove punctuation
df_1_filter['body_clean'] = df_1_filter['body'].apply(lambda x: re.sub('[!"#$“%&*”+,-.’/―)(:;<=>?@[\]^_`{|}~]', '', x))
# Convert the titles to lowercase
df_1_filter['body_clean'] = df_1_filter['body_clean'].apply(lambda x: x.lower())


# The content of the articles is searched for the words terrorism and terrorist using regular expressions. 

# In[5]:


string_list = ['terrorism', 'terrorist']


# In[6]:


def find_match_count(word: str, pattern: str) -> int:
    return len(re.findall(pattern, word.lower()))


# In[7]:


for col in string_list:
    df_1_filter[col] = df_1_filter['body_clean'].apply(find_match_count, pattern=col)


# In[8]:


#Only keep the rows where either the word terrorism or terrorist occurs once or more.
terrorisme1 = df_1_filter[(df_1_filter.terrorism > 1) | (df_1_filter.terrorist >1)]


# ### Dataset part 2

# In[9]:


df_2 = pd.read_csv('GGT_deel2.csv')


# In[10]:


df_2_filter = df_2[['body', 'outlet' , 'publishedAt', 'people', 'authorsFemaleCount', 'authorsMaleCount','authorsUnknownCount', 'peopleCount', 'peopleFemaleCount', 'peopleMaleCount',
       'peopleUnknownCount','sourcesFemale', 'sourcesFemaleCount', 'sourcesMale',
       'sourcesMaleCount', 'sourcesUnknown', 'sourcesUnknownCount' , 'voicesFemale',
       'voicesMale']]


# In[11]:


# Remove punctuation
df_2_filter['body_clean'] = df_2_filter['body'].apply(lambda x: re.sub('[!"#$“%&*”+,-.’/―)(:;<=>?@[\]^_`{|}~]', '', x))
# Convert the titles to lowercase
df_2_filter['body_clean'] = df_2_filter['body_clean'].apply(lambda x: x.lower())


# The content of the articles is searched for the words terrorism and terrorist using regular expressions. 

# In[12]:


string_list = ['terrorism', 'terrorist']


# In[13]:


def find_match_count(word: str, pattern: str) -> int:
    return len(re.findall(pattern, word.lower()))


# In[14]:


for col in string_list:
    df_2_filter[col] = df_2_filter['body_clean'].apply(find_match_count, pattern=col)


# In[15]:


#Only keep the rows where either the word terrorism or terrorist occurs once or more.
terrorisme2 = df_2_filter[(df_2_filter.terrorism > 1) | (df_2_filter.terrorist >1)]


# ### Dataset part 3

# In[16]:


df_3 = pd.read_csv('GGT_deel3.csv')


# In[17]:


df_3_filter = df_3[['body', 'outlet' , 'publishedAt', 'people', 'authorsFemaleCount', 'authorsMaleCount','authorsUnknownCount', 'peopleCount', 'peopleFemaleCount', 'peopleMaleCount',
       'peopleUnknownCount','sourcesFemale', 'sourcesFemaleCount', 'sourcesMale',
       'sourcesMaleCount', 'sourcesUnknown', 'sourcesUnknownCount' , 'voicesFemale',
       'voicesMale']]


# In[18]:


# Remove punctuation
df_3_filter['body_clean'] = df_3_filter['body'].apply(lambda x: re.sub('[!"#$“%&*”+,-.’/―)(:;<=>?@[\]^_`{|}~]', '', x))
# Convert the titles to lowercase
df_3_filter['body_clean'] = df_3_filter['body_clean'].apply(lambda x: x.lower())


# The content of the articles is searched for the words terrorism and terrorist using regular expressions. 

# In[19]:


string_list = ['terrorism', 'terrorist',]


# In[20]:


def find_match_count(word: str, pattern: str) -> int:
    return len(re.findall(pattern, word.lower()))


# In[21]:


for col in string_list:
    df_3_filter[col] = df_3_filter['body_clean'].apply(find_match_count, pattern=col)


# In[22]:


#Only keep the rows where either the word terrorism or terrorist occurs once or more.
terrorisme3 = df_3_filter[(df_3_filter.terrorism > 1) | (df_3_filter.terrorist >1)]


# In[23]:


#Merge the frames together in 1 big frame. 
frames = [terrorisme1, terrorisme2, terrorisme3]
terrorism_complete = pd.concat(frames)


# In[24]:


#Save the merged dataset to a csv file
#terrorism_complete.to_csv('terrorism_complete.csv')

