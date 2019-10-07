#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv as csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth',300)


# In[2]:


path=r'C:\Users\rohan\test\MovieSummaries\MovieSummaries\movie.metadata.tsv'
meta=pd.read_csv(path,sep='\t',header=None)
meta.head()


# In[3]:


meta.head()
print(type(meta))


# In[4]:


#meta.colums=['movie_id',1,'movie_name',3,4,5,6,7,'genre']
meta.rename(columns={0:'movie_id',2:'movie_name',8:'genre'}, inplace=True)
meta.head()


# In[ ]:





# In[5]:


plots = []

with open("plot_summaries.txt", 'r',encoding="utf8") as f:
    reader = csv.reader(f, dialect='excel-tab', delimiter='\t') 
    for row in tqdm(reader):
        plots.append(row)


# In[ ]:





# In[6]:


print(plots[0])


# In[7]:


movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in plots:
    movie_id.append(i[0])
    plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})


# In[8]:


movies.head()


# In[11]:


# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

meta = meta.reindex(columns=['movie_id', 'movie_name', 'genre'])
# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')
movies.head()


# In[12]:


movies['genre'][0]


# In[13]:


json.loads(movies['genre'][0]).values()


# In[14]:


genres=[]
for i in movies['genre']:
    genres.append(list(json.loads(i).values()))
    
movies['genre_new']=genres


# In[15]:


movies_new = movies[~(movies['genre_new'].str.len() == 0)]


# In[16]:


movies_new.shape, movies.shape


# In[17]:


movies.head()


# In[18]:


all_genres=sum(genres,[])
len(set(all_genres))


# In[19]:


all_genres=nltk.FreqDist(all_genres)
all_genres_df=pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})


# In[20]:


g=all_genres_df.nlargest(columns="Count",n=20)
plt.figure(figsize=(12,15))
ax=sns.barplot(data=g,x="Count",y="Genre")
ax.set(ylabel='Count')
plt.show()


# In[21]:


def clean_text(text):
    text = re.sub("\'", "", text)
    text=re.sub("[^a-zA-Z]"," ",text)
    text=' '.join(text.split())
    text=text.lower()
    return text


# In[22]:


movies_new['clean_plot']=movies_new['plot'].apply(lambda x: clean_text(x))


# In[23]:


movies_new.head()


# In[26]:


def freq_words(x, terms=30):
    all_words=' '.join([text for text in x])
    all_words=all_words.split()
    fdist=nltk.FreqDist(all_words)
    words_df=pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    d=words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(12,15))
    ax=sns.barplot(data=d,x="count",y="word")
    ax.set(ylabel='Word')
    plt.show()
freq_words(movies_new['clean_plot'],100)    


# In[27]:


nltk.download('stopwords')


# In[31]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
def remove_stopwords(text):
    no_step_text=[w for w in text.split() if not w in  stop_words]
    return ' '.join(no_step_text)
movies_new['clean_plot']=movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))


# In[33]:


freq_words(movies_new['clean_plot'],50)


# In[34]:


from sklearn.preprocessing import MultiLabelBinarizer
multilabel_binarizer=MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])
y=multilabel_binarizer.transform(movies_new['genre_new'])


# In[35]:


tfidf_vectorizer=TfidfVectorizer(max_df=0.8,max_features=10000)


# In[36]:


xtrain, xval, ytrain, yval=train_test_split(movies_new['clean_plot'],y,test_size=0.2,random_state=9)


# In[37]:


xtrain_tfidf=tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf=tfidf_vectorizer.transform(xval)


# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
lr=LogisticRegression()
clf=OneVsRestClassifier(lr)
clf.fit(xtrain_tfidf,ytrain)


# In[41]:


y_pred=clf.predict(xval_tfidf)


# In[42]:


y_pred[3]


# In[43]:


multilabel_binarizer.inverse_transform(y_pred)[3]


# In[44]:


f1_score(yval,y_pred,average="micro")


# In[45]:


y_pred_prob = clf.predict_proba(xval_tfidf)


# In[57]:


t = 0.2 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)
f1_score(yval, y_pred_new, average="micro")


# In[60]:


def infer_tags(q):
    q=clean_text(q)
    q=remove_stopwords(q)
    q_vec=tfidf_vectorizer.transform([q])
    q_pred=clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


# In[61]:


for i in range(5):
    k=xval.sample(1).index[0]
    print("Movie: ", movies_new['movie_name'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",movies_new['genre_new'][k], "\n")


# In[ ]:




