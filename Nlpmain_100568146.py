#!/usr/bin/env python
# coding: utf-8

# In[1]:



cd C:\\Users\\teenu\\Desktop\\Study materials\\NLP


# In[2]:


#Importing the dataset
import pandas as pd
data=pd.read_csv("fake_job_postings.csv")
data.info()


# In[3]:


#Finding the distribution of the data
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6, 4))
ax = sns.countplot(data.fraudulent)
plt.title('The distribution of the target feature (fraudulent)')
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.33, p.get_height()))

plt.show()


# In[4]:


#data exploration and preprocessing
f_data=data[data['fraudulent']==1]
r_data=data[data['fraudulent']==0]
f_data.shape,r_data.shape


# In[5]:


#getting equal amount of real data as of fake data
r_dat=r_data.sample(f_data.shape[0])
r_dat.shape


# In[6]:


#appending the balanced data into one dataframe
new_data=r_dat.append(f_data,ignore_index='True')
new_data['fraudulent'].value_counts()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6, 4))
ax = sns.countplot(new_data.fraudulent)
plt.title('The distribution of the target feature (fraudulent)')
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.33, p.get_height()))

plt.show()


# In[8]:


#removing the missing value.
updated_df = new_data.dropna(subset=['description'])
updated_df.info()


# In[9]:


#data preprocessing for NLP

c=[]
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords
ps=PorterStemmer()
for i in updated_df.index:
    n=re.sub('[^a-zA-Z]',' ', updated_df['description'][i])
    n=n.lower()
    n=n.split()
    n=[ps.stem(word) for word in n if word not in stopwords.words('english') ]
    n=" ".join(n)
    
    c.append(n)


# In[10]:


updated_df.description=c
updated_df.head()


# In[11]:


# Model Building
#model 1-----Random Forest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
classifier=Pipeline([("tfidf", TfidfVectorizer()) , ("classifier", RandomForestClassifier(n_estimators=100))])
X_train, X_test,y_train, y_test=train_test_split( updated_df['description'], updated_df['fraudulent'], test_size=0.30, random_state=0, shuffle=True)
y=updated_df.iloc[:, 6].values
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
accuracy_score(y_test, y_pred)


# In[12]:


#Model 2 -----Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(updated_df['description'], updated_df['fraudulent'], test_size=0.30, random_state=0, shuffle=True)
svm=Pipeline([("tfidf",TfidfVectorizer()),("classifier",SVC(C=100,gamma='auto'))])
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
accuracy_score(y_test, y_pred)


# In[13]:


#Model 3 ----Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=updated_df.iloc[:, 17].values
X_train, X_test,y_train, y_test=train_test_split(x,y, test_size=0.30, random_state=0, shuffle=True)
classi=GaussianNB()
classi.fit(X_train, y_train)
y_pred=classi.predict(X_test)
accuracy_score(y_test, y_pred)


# In[14]:


# Now the variables company_profile, description and requirements are joined together and 
#the same models are performed and checked their performances.


# In[15]:


# Joining the data
data.head()


# In[16]:


new_data=r_dat.append(f_data,ignore_index='True')
new_data.shape


# In[17]:


#removing the missing value.
updated_df1= new_data.dropna(subset=['description','company_profile','requirements'])
updated_df.info()


# In[18]:


#Joining the datset
updated_df1['full']=new_data['company_profile'] + new_data['description']+new_data['requirements']


# In[19]:


updated_df1=updated_df1[['full','fraudulent']]
updated_df1.head()


# In[20]:


#Removing the missing values
updated_df1 = updated_df1.dropna(subset=['full'])
updated_df1.info()


# In[21]:


#Prepping for NLP
c=[]
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords
ps=PorterStemmer()
for i in updated_df1.index:
    n=re.sub('[^a-zA-Z]',' ', updated_df1['full'][i])
    n=n.lower()
    n=n.split()
    n=[ps.stem(word) for word in n if word not in stopwords.words('english') ]
    n=" ".join(n)
    
    c.append(n)


# In[22]:


updated_df1.full=c
updated_df1.head()


# In[23]:


# Model Building
#model 1-----Random Forest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
classifier=Pipeline([("tfidf", TfidfVectorizer()) , ("classifier", RandomForestClassifier(n_estimators=100))])
X_train, X_test,y_train, y_test=train_test_split( updated_df1['full'], updated_df1['fraudulent'], test_size=0.30, random_state=0, shuffle=True)
y=updated_df.iloc[:, 6].values
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
accuracy_score(y_test, y_pred)


# In[24]:


#Model 2 -----Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(updated_df1['full'], updated_df1['fraudulent'], test_size=0.30, random_state=0, shuffle=True)
svm=Pipeline([("tfidf",TfidfVectorizer()),("classifier",SVC(C=100,gamma='auto'))])
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
accuracy_score(y_test, y_pred)


# In[25]:


#Model 3 ----Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=updated_df1.iloc[:, 1].values
X_train, X_test,y_train, y_test=train_test_split(x,y, test_size=0.30, random_state=0, shuffle=True)
classi=GaussianNB()
classi.fit(X_train, y_train)
y_pred=classi.predict(X_test)
accuracy_score(y_test, y_pred)


# In[26]:


#Word Embedding
import nltk
nltk.download('punkt')


# In[27]:


import nltk
import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors


# In[28]:


des=updated_df['description'].values
des[:3]


# In[29]:


#Tokenize the description
des_tok=[nltk.word_tokenize(description) for description in des]


# In[30]:


des_tok[0]


# In[31]:


#model building
model=Word2Vec(des_tok, min_count=1)
model


# In[32]:


#Predict the Word2Vec 
model.wv['team']


# In[33]:


#Finding the similar word
model.wv.most_similar('work')


# In[34]:


#Convolutional Neural Network
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Embedding
from tensorflow.keras.models import Model


# In[35]:


df=updated_df[['description','fraudulent']]
df.info()


# In[36]:


X_train, X_test,y_train, y_test=train_test_split(df['description'],df['fraudulent'], test_size=0.30)


# In[37]:


#converting into sequence
max_vocab_size=100000
tokenizer=Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(X_train)
sequences_train=tokenizer.texts_to_sequences(X_train)
sequence_test=tokenizer.texts_to_sequences(X_test)


# In[38]:


#unique tokens
word2idx=tokenizer.word_index
v=len(word2idx)
v


# In[39]:


#pad sequences for equal sequences
data_train=pad_sequences(sequences_train)
t=len(data_train[0])


# In[40]:


#pad the test datset
data_test=pad_sequences(sequence_test,maxlen=t)
len(data_test[1])


# In[41]:


#Building the model
D=30
i=Input(shape=(t,) )
x=Embedding(v+1, D)(i)
x=Conv1D(32,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=GlobalMaxPooling1D()(x)
x=Dense(1,activation='sigmoid')(x)
model=Model(i,x)


# In[42]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accurancy'])
model.compile(loss='binary_crossentropy', metrics=['accuracy'])


# In[43]:


#train the model
r=model.fit(x=data_train, y=y_train,epochs=5,validation_data=(data_test, y_test))


# In[44]:


#The loss function
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='Loss')
plt.plot(r.history['val_loss'],label='Validation_ loss')
plt.legend()
plt.show()


# In[45]:


#The accuracy
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'],label='Accuracy')
plt.plot(r.history['val_accuracy'],label='Validation_ accuracy')
plt.legend()
plt.show()


# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM,Embedding
from tensorflow.keras.models import Model


# In[47]:


#Recurrent Neural Network
#Building the model
D=30
M=15
i=Input(shape=(t,) )
x=Embedding(v+1, D)(i)
x=LSTM(M,return_sequences=True)(x)
x=MaxPooling1D()(x)

x=Dense(1,activation='sigmoid')(x)
model=Model(i,x)


# In[48]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accurancy'])
model.compile(loss='binary_crossentropy', metrics=['accuracy'])


# In[49]:


#train the model
r=model.fit(x=data_train, y=y_train,epochs=10,validation_data=(data_test, y_test))


# In[50]:


#Text Summerization


# In[51]:


pip install -U spacy


# In[52]:


get_ipython().system('python -m spacy download en')


# In[53]:


#Text Summary


# In[54]:


len(data.description[1])


# In[55]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import en_core_web_sm

nlp = en_core_web_sm.load()


# In[56]:


doc=nlp(data.description[1])


# In[57]:


tokens=[token.text for token in doc]
print(tokens)


# In[58]:


punctuation=punctuation + '\n'
print(punctuation)


# In[59]:


#Word Frequency
word_freq={}
stop_words=list(STOP_WORDS)
for word in doc:
   if word.text.lower() not in stop_words:
        if word.text.lower() not in punctuation:
          if word.text not in word_freq.keys():
            word_freq[word.text]=1
          else:
            word_freq[word.text]+= 1

            
print(word_freq)


# In[60]:


max_freq=max(word_freq.values())


# In[61]:


for word in word_freq.keys():
    word_freq[word]=word_freq[word]/ max_freq
    
print(word_freq)


# In[62]:


# sentence tokenization
sent_toc=[sent for sent in doc.sents]
print(sent_toc)


# In[63]:


sent_score={}
for sent in sent_toc:
    for word in sent:
        if word.text.lower() in word_freq.keys():
            if sent not in sent_score.keys():
                sent_score[sent]=word_freq[word.text.lower()]
            else:
                sent_score[sent] += word_freq[word.text.lower()]
print(sent_score)


# In[64]:


#getting 30% of the total number of words. 
from heapq import nlargest
len(sent_score)*0.3


# In[65]:


#Getting the 5 highest scored sentences
reduced=nlargest(n=5,iterable=sent_score,key=sent_score.get)


# In[66]:


#the summary
final_summary=[word.text for word in reduced]
"".join(final_summary)


# In[ ]:




