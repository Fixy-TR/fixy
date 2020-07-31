# -*- coding: utf-8 -*-


import pandas as pd
df = pd.read_csv('KarışıkDuygular.csv')
df.tail(15)

del df['ID']

# kategorileri integer değerlere map ediyoruz, sınıflandırmanın yapılabilmesi için
df["category_id"] = df.OriginalEmotion.map({'Happy':0,'Disgust':1,'Surprise':2,'Fear':3,'Anger':4,'Sadness':5})
df.head(28)

# stopwordleri çıkarıyoruz
with open('stopwords-tr.txt', 'r') as f:
    myList = [line.strip() for line in f]

print(df.Entry)

# cümleleri vektörize ediyoruz 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=myList)

features = tfidf.fit_transform(df.Entry).toarray()
labels = df.OriginalEmotion
features.shape

# unigram ve bigramları çıkarıyoruz
from io import StringIO
category_id_df = df[['OriginalEmotion', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'OriginalEmotion']].values)


from sklearn.feature_selection import chi2
import numpy as np

N = 10
for OriginalEmotion, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(OriginalEmotion))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

# 4 farklı model deniyoruz, en yüksek accuracyi LinearSVC ile alıyoruz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

cv_df.groupby('model_name').accuracy.mean()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.10, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)




# accuracy
accuracy_score(y_test, y_pred)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


model2 =  LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.10, random_state=0)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)

# accuracy
accuracy_score(y_test, y_pred)

# test cümleleri oluşturuyoruz
texts = ["kibirli olacağım bir anda ölçüt olarak kullanmamın icap edeceği bir listeyi oluşturacak muhteşem insanlardır.",
         "çok mutluyum",
         "sana çok kızgınım",
         "beni şaşırttın",
         "şu an ağlıyorum"]
text_features = tfidf.transform(texts)
predictions = model2.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  print("")

#modelimizi kaydediyoruz
import pickle
filename = 'emotion_model.pickle'
pickle.dump(model, open(filename, 'wb'))

# modeli yükleyip test ediyoruz
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=myList)

loaded_model = pickle.load(open("emotion_model.pickle", 'rb'))
corpus = [
           'mutlu.',
         'mutluyum',
        'mutlu',
        'mutlu',
        ]
tfidf.fit_transform(df.Entry).toarray()
features = tfidf.transform(corpus).toarray()
result = loaded_model.predict(features)
print(result)
