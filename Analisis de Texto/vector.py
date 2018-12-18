import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
#nltk.download()
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

df = pd.read_csv('C:/Users/asus/Desktop/proyecto/textdata.csv')
df = df[['number','text','value']]
df = df[pd.notnull(df['text'])]
df.rename(columns = {'text':'text'}, inplace = True)
df.head(10)

print(df.shape)

########################Grafica de datos################
df.index = range(192)
#df['text'].apply(lambda x: len(x.split(' '))).sum()
cnt_pro = df['value'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('value', fontsize=12)
plt.xticks(rotation=90)
plt.show();
##########################################################

df['text'] = df['text'].apply(cleanText) #vamos a preprocesar los textos

train, test = train_test_split(df, test_size=0.3, random_state=42)# vamos a separar en dos bases de datos para el entrenamiento

#tokenizar texto utilizando el tokenizador NLTK, en nuestro primer intento etiquetamos cada texo con su sentimeinto.
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.value]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.value]), axis=1)

#print(train_tagged.values[3])#ver los keywords etiquetados con su respectiva etiqueta positivo o negativo

###############################################Crear el modelo paragraph vector DM########################
model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

y_train, X_train = vec_for_learning(model_dmm, train_tagged)
y_test, X_test = vec_for_learning(model_dmm, test_tagged)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))