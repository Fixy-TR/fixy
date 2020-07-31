# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,  Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# Wikipedia verisini okuyoruz, temizliyoruz ve dataframe atıyoruz

import pandas as pd
with open('/content/drive/My Drive/wiki_00', 'r') as file:
    datam = file.read().replace('\n', '')
arr = datam.split('. ')
df1 = pd.DataFrame(arr)
df1 = df1[~df1[0].str.contains("<", na=False)]
df1.head()

df1.shape

# verinin ilk 1200000'lik kısmını alıyoruz ama hepsi ile eğitirsek daha doğru sonuçlar verebilir
df1 = df1[:1200000]
df1["text"]=df1[0]
df1["label"]=1

df1 = df1.drop(columns=0)
df1.head()

df1.shape

# Farklı konulardaki haber verilerini de ayırdığımız wikipedi verilerine ekliyoruz
data1 = pd.read_csv('7all.csv', encoding='utf-8', header=None, names=['cat', 'text'])
data1=data1.drop(["cat"], axis=1)
data1.tail()

# bu veriyi 1 ile etiketiliyoruz ve bunlar formal verilerimiz oluyor
data1["label"]= 1 #formal

data1.shape

data1.tail()

df1=data1.append(df1)

df1.to_csv("Formal_Data.csv")

df1.shape

# Twitter verisini okuyoruz ve dataframede tutuyoruz
data = pd.read_csv('sentimentdata.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
#data.sentiment = data.sentiment.map({'positive':0,'negative':1})
data.head()

data.shape

#işimize yaramayacak columnları çıkarıyoruz
data = data.drop(columns="static_link")
data = data.drop(columns="retweets")
data = data.drop(columns="favorites")
data = data.drop(columns="class")
data = data.drop(columns="created_at")
data = data.drop(columns="user_name")

#Twitter verisini 0 ile etiketliyoruz ve informal verimizi elde etmiş oluyoruz
data["label"]=0

data.head()

#Twitter verisinin 1300000'lük kısmını alıyoruz
data = data[:1300000]

data.shape

# normalizasyon yaparak verilerimizi temizlemek için fonksiyonlar
def extract_emojis(str):
    return [c for c in str if c in emoji.UNICODE_EMOJI]
def sentiment_emojis(sentence):
    emojis = extract_emojis(sentence)
    result = [0,0,0,0]
    if len(emojis) == 0:
        return result
    for icon in emojis:
        sen_dict = analyzer_emoji.polarity_scores(icon)
        sen = [sen_dict['neg'],sen_dict['neu'],sen_dict['pos'],sen_dict['compound']]
        result = [result[i] + sen[i] for i in range(4)]
    return [result[i] / len(emojis) for i in range(4)]
def sentiment_emojis_row(row):
    comment = row['text']
    sen_comment = sentiment_emojis(comment)
    
    row['emoji_neg'] = sen_comment[0]
    row['emoji_neu'] = sen_comment[1]
    row['emoji_pos'] = sen_comment[2]
    row['emoji_compound'] = sen_comment[3]
    
    return row

df = data.copy()

# verilerimizi temizliyoruz
import re
'''NLTK Porter Stemmer da kullanılabilir'''

df['text'] = df['text'].astype(str).fillna(' ')
# Lower case comment
df['text'] = df['text'].str.lower()
# Add num words of comment as feature
#df['num_words'] = df['text'].apply(lambda s: len(s.split()))
# Add num words unique of comment as feature
#df['num_unique_words'] = df['text'].apply(lambda s: len(set(w for w in s.split())))
# Add num words unique per num words of comment as feature
#df['words_vs_unique'] = df['num_unique_words'] / df['num_words'] * 100
# Add emojis features
print("Statistical features end!")
def remove_urls(text):
  pattern = re.compile(r'https?://\S+|www\.\S+')
  return pattern.sub(r'', text)
df['text']= df['text'].apply(remove_urls)

# formal ve informal verileri birleştiriyoruz
data_new = df.append(df1)

data_new.shape

# get all labels and reviews as a list
target = data_new['label'].values.tolist()#
datas = data_new['text'].astype(str).tolist()#text data

#verimizi test ve train olarak bölüyoruz
seperation = int(len(datas) * 0.80)
x_train, x_test = datas[:seperation], datas[seperation:]
y_train, y_test = target[:seperation], target[seperation:]

# verisetinde en sık geçen 10000 kelimeyi alıyoruz
num_words = 10000 

# keras ile tokenizer tanımlıyoruz
tokenizer = Tokenizer(num_words=num_words)

# veriyi tokenize ediyoruz
tokenizer.fit_on_texts(datas)

# tokenizerı kaydediyoruz
import pickle

with open('tokenizer_FormalInformal.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# tokenizerı yüklüyoruz
import pickle

with open('tokenizer_informal.pickle', 'rb') as handle:
    turkish_tokenizer = pickle.load(handle)

# train verisini tokenize ediyoruz
x_train_tokens = tokenizer.texts_to_sequences(x_train)

# test verisini tokenize ediyoruz
x_test_tokens = tokenizer.texts_to_sequences(x_test)

# RNN modellerine önceden tanımlanmış sabit boyutlu veriler verebiliriz, bu yüzden diğer verilerin boyutlarını en büyük veriye tamamlamak için padding yapıyoruz


# her veride kaç token var bakıyoruz
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]

# listeyi numpy arraye çeviriyoruz
num_tokens = np.array(num_tokens)
num_tokens.shape

# bir veride ortalama kaç kelime var
np.mean(num_tokens)

# maximum kelime miktarı
np.max(num_tokens)

# maximum kelimenin bulunduğu index
np.argmax(num_tokens)

# bütün veriler için maximum token sayısı
max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens) #returns float
max_tokens = int(max_tokens)
max_tokens

np.sum(num_tokens < max_tokens) / len(num_tokens) * 100

# Let's add paddings... So, all datas will be in the same size.
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

# test size ve train size
print(x_train_pad.shape)
print(x_test_pad.shape)

model = Sequential() # modeli tanımlıyoruz

embedding_size = 50 # her kelime için 50 size belirliyoruz

#kerasta embedding layer oluşturuyoruz
#rastgele kelime vektörleri oluşturuyoruz
# embedding layerı modele ekliyoruz
# embedding matris size = num_words * embedding_size -> 10.000 * 50
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer'))

# 3-layered LSTM
model.add(LSTM(units=16, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=8, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=4, return_sequences=False))
model.add(Dropout(0.2))
# Dense layer: fully connected layer
model.add(Dense(1, activation='sigmoid'))

from tensorflow.python.keras.optimizers import Adam
# Adam optimizer
optimizer = Adam(lr=1e-3)

# farklı optimizerlar deniyoruz
model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])

model.summary()

# epoch -> veriyle kaç kere eğiteceğiz
# batch_size -> her epochta ne kadar veri ile modeli besleyeceğiz
model.fit(x_train_pad, y_train, epochs=5, batch_size=256)

result = model.evaluate(x_test_pad, y_test)# model test verisini etiketliyor
result

accuracy = (result[1]) * 100
accuracy # modelin doğruluğu

# test inputları oluşturuyoruz
text1 = "atatürk, bu görevi en uzun süre yürüten kişi olmuştur."
text2="bdjfhdjfhdjkhj"
text3 = "hiç resimde gösterildiği gibi değil..."
text4 = "bir yirminci yüzyıl popüler kültür ikonu haline gelen ressam, resimlerinin yanı sıra inişli çıkışlı özel yaşamı ve politik görüşleri ile tanınır. "
text5 = "fransız halkı önceki döneme göre büyük bir evrim geçirmektedir. halk bilinçlenmektedir ve sarayın, kralın, seçkinlerin denetiminden çıkmaya başlamıştır. şehirlerde yaşayan pek çok burjuva, büyük bir atılım içindedir. kitaplar yaygınlaşmakta, aileler çocuklarını üniversitelere göndererek sağlam bir gelecek kurma yolunu tutarak kültürel seviyeyi yükseltmektedir. bağımsız yayıncıların çıkardıkları gazete, bildiri ve broşürler, kitlesel bilinçlenmeye yol açmaktadır. bu koşullar da toplumsal değişim taleplerinin olgunlaşmasına yol açmıştır.Devrimden önceki yıllar Fransız ekonomisi için pek de parlak sayılmamaktadır. Gelişen ticaret, savaşlar sebebiyle yavaşlama yönüne kaymış; köylü, mahsulünden beklenen verimi alamayarak büyük sıkıntılarla karşılaşmıştır. Ayrıca, tek kıtlıkla, açlığa kadar dayanan sorunlarla karşılaşmışlar tek çözüm yolu olarak kıta şehirlere göç etme yolunu tutmuşlardır, fakat şehirlerde de onları parlak bir yaşam beklememektedir; artan nüfusun ihtiyacını şehirler karşılayamaz duruma gelmiştir. Nüfus artması doyurulması gereken insanların çoğalmasına sebep olmuştur. Gelenlerin işsizlik sorunuyla da karşılaşması, istihdam olanağı bulamamaları toplumsal sorunların artmasına neden olmuştur. Aslında Fransa’nın ekonomisi pek çok çağdaş devlete göre ileri sayılmaktaydı; fakat önceki dönemlerle karşılaştırıldığında görülen fark edilir gerileme, halkı paniğe sokmuştur.[1] Halkın içinde bulunduğu ekonomik sorunlar vergilerin düzenli olarak ödenmemesine yol açmış devletin en önemli gelir kaynağı olan vergilerin sekteye uğraması hazineyi büyük bir bunalıma sürüklemiş, uzayan savaş maliyetlerinin fazla olması ve teknolojinin gelişmesiyle savaş masraflarının artması, bir de saray masraflarının aşırılığı sebebi ile devlet iflasın eşiğine gelmiştir. Bu nedenle kral, vergilerin artırılması ve yeni vergiler konması yolunu tutmuş; bu plan dahilinde tüm toplumda vergilerin yaygınlaşması düşüncesi ortaya çıkmıştır. Paris Parlamentosu da bu yeni vergi aleyhine onay vermeyerek genel meclisin, Etats Generaux'un toplanmasını istemiştir."
text6 = "bunu çıkardım söylediklerinden"
text7 = "Bu koşullar da toplumsal değişim taleplerinin olgunlaşmasına yol açmıştır."
text8="bu çok saçma yaa"
text9="bana böyle bir yetki verilmedi."
text10="napıcaz bu işi böyle"
text11="Öncelikle Mercedes-Benz’e olan ilgin için teşekkür ederiz."
text12="Ekibimizle çalışma isteğin için teşekkür ediyor, sağlıklı günler ve kariyerinde başarılar diliyoruz. Farklı etkinlik ve programlarda tekrar bir araya gelmek dileğiyle."
text13="Ben de öyle olduğunu düşünmüyordum ama gittik yine de jzns"
texts = [text1, text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13]

tokens = tokenizer.texts_to_sequences(texts)

tokens_pad = pad_sequences(tokens, maxlen=max_tokens)

model.predict(tokens_pad)# 0(informal) ve 1(formal) arasında değerler verecek

#test verisini tahminleme
for i in model.predict(tokens_pad):
  if i < 0.5:
    print("informal")
  elif i >= 0.5:
    print("formal")

# modeli kaydediyoruz
model.save("MODEL_FORMAL.h5")
print("Saved model to disk")

'
from keras.models import load_model
# modeli yüklüyoruz
model = load_model('moddel.h5')
