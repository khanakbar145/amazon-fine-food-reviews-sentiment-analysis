#!/usr/bin/env python
# coding: utf-8

# ## Amazone fine food by Khan Akbar

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import sqlite3
import pandas as pd 
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd


# In[2]:


# using the sqllite to read the data
con = sqlite3.connect(r"C:\Users\khana\Downloads\amazon-fine-food-reviews/database.sqlite")


# In[3]:


import numpy as np
import math
import pandas as pd
## filtring the only positive and negative reviews i.e not taking the reviews with score =
filtered_data = pd.read_sql_query("""
SELECT * 
FROM Reviews 
WHERE Score != 3
""", con) # con is a link between sql data base to our code
#Given review with score >3 a positve rating and review with a score greater then 3
def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'
#changing the reviews with score less than 3 to negative  and gratear than 3 to positive by using the "score" colum in the data set 
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative
#filtered


# In[4]:


filtered_data.shape #looking at the number of attributes and size of the data set 
filtered_data.head()
## the score is now converted into positive and negative 


# ## Data Cleaning :Deduplication 
# it is observed that review data has many duplication entries Hence it was necessary to remove duplication in order to get unbiased result for the analysis of the data . following is an example 

# In[5]:


display= pd.read_sql_query("""
SELECT * 
FROM Reviews 
WHERE Score != 3  AND UserID="AR5J8UI46CURR"
ORDER BY ProductID
""", con)
display


# ## in the above summary 
# we have seen that multiple review of the same values for ProfileName	HelpfulnessNumerator	HelpfulnessDenominator	Score	Time	Summary	Text product id was the same hence in order to reduce the redundancy it was decided to eliminate the rows having the same paremeters . 
# 
# the methods use the same was we first sort the data according to productid and then just keep the similar and delelte the other for eg. in the above just review for productid BOOOHDL1RQ this methods insuar that this methods is only one representatives still existing for the same product.  
# 

# In[6]:


#Shorting data accoriding the productid in ascending order 
sorted_data=filtered_data.sort_values('ProductId', axis=0,ascending=True,)


# In[7]:


#Deduplication of entries 
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep="first", inplace=False)
final.shape


# ## drop_duplicates 
# this function are using to remove the duplicate valu in this data set 
# **DataFrame.drop_duplicates(subset=None,keep="first",inplace=False
# Return data frame with duplicate rows removed only considering certain columns
# 

# In[8]:


#checking how mutch % data are still remains 
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# ## after removing the duplicate vaile 69.25% data are remaning from the total data set 

# ## Observation 
# it was seen that the two rows given below the value of HelpfulnessNumeratoris greater than HelpfulnessDenominator which is not practically possible Because HelpfulnessNumeratoris=yes,HelpfulnessDenominator=yes+no.hence we remove this tupes of column

# In[9]:


display= pd.read_sql_query("""
SELECT * 
FROM Reviews 
WHERE Score != 3  AND Id=44737 OR Id=64422
ORDER BY ProductID
""", con)
display


# In[10]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[11]:


#Before startinf the next phase of preprocessing lets see the number of enteries 
print(final.shape)

# How many positive and negative reviews are resent in the data set 
final['Score'].value_counts()


# ## Bag of Words(BoW)

# In[12]:


#Bow
count_vect = CountVectorizer() #in sckit-learn
final_counts = count_vect.fit_transform(final['Text'].values)


# In[13]:


type(final_counts)


# In[14]:


final_counts.get_shape()


# ## Text Preprocessing :Stemming,stop-word removal and Lemmatization.

# In[15]:


#find sentences containing HTML tages
import re
i=0;
for sent in final['Text'].values:
    if (len(re.findall('<. * ?>' , sent))):
        print(i)
        print(sent)
        break;
        
    i +=1;
        


# In[16]:


from nltk.corpus import brown
import nltk
#nltk.download()
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stop = set(stopwords.words('english')) #set the stopwords
sno = nltk.stem.SnowballStemmer('english') #initailistion of sonwball

def cleanhtml(sentence): #function to clean the word of any html tages 
    cleanr = re.compile('<.*?')
    cleantext = re.sub(cleanr,' ', sentence)
    return cleantext

def cleanpunc(sentence): # function to clean the word of any puncation 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]' ,r' ',cleaned)
    return cleaned
print(stop)
print('******************************************')
print(sno.stem('tasty'))
    


# In[ ]:


# code for implementing step-by-step the check mentioned
#this code takes a while to run as it need to run 
import os 
i=0
str1=' '
final_string=[]
all_positive_words=[] #store word form positive reviwe 
all_negative_words=[] #store word form negative review
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    print (sent);
    sent=cleanhtml(sent) # remove HTML tages
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i]=='postive':
                        all_positive_words.append(s)
                    if(final['Score'].values)[i]=='negative':
                        all_negative_words.append(s) # list of all negative words
                else:
                    continue
            else:
                continue
        print(filtered_sentence)
    str1 = b" ".join(filtered_sentence)# final string of cleaned words
    #print("******************************************************")
    final_string.append(str1)
    i+=1
    
                            


# In[ ]:


final['CleanedText']=final_string # adding a column of cleantext which 
final.head(3) #below the processing review can be seen in the cleantext


#store final tabel into an SQLLITE tabel for future,
conn = sqlite3.connect('final.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews',conn,flavor=None,schema=None,if_exists='replace')


# ## Bi-Grams and n-Grams.
# we have to describe positive and negative reviews lets analyse them.
# using frequence distribution of the words as shown below

# In[ ]:


freq_dist_positive=nltk.FreqDist(all_positive_words)
freq_dist_negative=nltk.FreqDist(all_negative_words)
print("Most Common Positive Words": ,freq_dist_positive.most_common(20))
print("Most Common Negative Words": ,freq_dist_negative.most_common(20))


# ## Observation
# From the above it can seen that the most common positive and negative words overlap for eg. 'like' could be use as 'not like'
# so it good idea for  consider pair of sequence word (bi-gram) or n-consetive (n-gram)

# In[ ]:


#bi-gram ,tri-gram and n-gram
#removing stop word like "not" should be avoided before building n-grames
count_vect = CountVectorizer(ngram_range=(1,2) ) # in sckit learn
final_bigram_counts = count_vect.fit_transform(final['Text'].values)


# In[ ]:


final_bigram_counts.get_shape()


# ## TF-IDF

# In[ ]:


tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
final_tf_idf = tf_idf_vect.fit_transform(final['Text'].values)


# In[ ]:





# In[ ]:


final_tf_idf.get_shape()


# In[ ]:


features = tf_idf_vect.get_feature_names()
len(features)


# In[ ]:


features[100000:100010]


# In[ ]:


# converat a row to sapresmatrix to a numpy array
print(final_tf_idf[3,:].toarray()[0])


# ## Word 2 vec

# In[ ]:


# Train your own word2vec model using your own text corpus
import gensim
i=0
list_of_sent=[]
for sent in final['Text'].values:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):
                filtered_sentence.append(cleaned_words.lower())
                else:
                    continue
                    list_of_sent.append(filtered_sentence)
                    


# In[ ]:


print(final['Text'].values[0])
print("**********************************************************")
print(list_of_sent[0])


# In[ ]:


w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=30,worker=4)


# In[ ]:


words = list(w2v_model.wv.vocab)
print(len(words))


# ## Avg W2V,TFIDF-W2V

# In[ ]:


# average w2v 
# compute word2vec to each review
sent_vectors = [];
for sent in list_of_sent:
    sent_vec = np.zeros(30)#for word vector eatch zero length
    cnt_words =0;#num of words with a valid vector in sentence
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            sent_vec +=vec
            cnt_words+=1
            except:
                pass
            
            sent_vec /= cnt_words
            sent_vectors.append(sent_vec)
            print(len(sent_vectors))
            print(len(sent_vectors[0]))
            


# In[ ]:


#TF-IDF weighted Word2Vec
tfidf_feat = tf_idf_vect.get_feature_names()
#the final tfidf is a sparse matrix with row=sentence,col=word and cell_value
tfidf_sent_vectors = [];
row=0;
for sent in list_of_sent:
    sent_vec=np.zeros(50)#as word vect are zero length
    weight_sum =0; #sum of words a valid vector in the sentance
    for word in sent:#for each word in review
        try:
            vec = w2v_model.wv[word]
            #option the tf-idf in a sentences /review
            tfidf = final_tf_idf[row,tfidf_feat.index(word)]
            sent_vec+=(vec*tf_idf)
            weight_sum+= tf_idf
            except:
                pass
            sent_vec/=weight_sum
            tfidf_sent_vectors.append(sent_vec)
            row += 1
            
            
            
            
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




