# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('./drive/My Drive/')

# veriyi pandas ile okuyoruz
data = pd.read_csv("sentiment_data.csv")

df = data.copy()
df.head()

#0->negatif veri etiketi
#1->pozitif veri etiketi
df['Rating'].unique().tolist()

#model için gerekli kütüphaneleri import ediyoruz
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,  Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# bütün verileri ve etiketleri listeye çeviriyoruz
target = df['Rating'].values.tolist()#negatif=0, pozitif=1
data = df['Review'].values.tolist()#text verisi

#veriyi test ve train verisi olarak ayırıyoruz
seperation = int(len(data) * 0.80)
x_train, x_test = data[:seperation], data[seperation:]
y_train, y_test = target[:seperation], target[seperation:]

#veri satır ve sütun sayısı
df.shape

# Verisetimizde en sık geçen 10000 kelimeyi alıyoruz
num_words = 10000

# Keras ile tokenizer tanımlıyoruz
tokenizer = Tokenizer(num_words=num_words)

# Veriyi tokenlara ayırıyoruz
tokenizer.fit_on_texts(data)

# Tokenizerı kaydediyoruz
import pickle

with open('turkish_tokenizer_hack.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Tokenizerı yüklüyoruz
with open('turkish_tokenizer_hack.pickle', 'rb') as handle:
    turkish_tokenizer = pickle.load(handle)

# Train verisi olarak ayırdığımız veriyi tokenizer ile tokenize ediyoruz
x_train_tokens = turkish_tokenizer.texts_to_sequences(x_train)

x_train[100]

x_train_tokens[100]

# Test verisi olarak ayırdığımız veriyi tokenizer ile tokenize ediyoruz

x_test_tokens = turkish_tokenizer.texts_to_sequences(x_test)

#Text verileri için padding yapıyoruz
#RNN ağlarını kullanırken önceden belirdiğimiz sabit bir size olur. Tüm input textlerinin sizelarını bu sabit size için padding yaparak 0 lar
#ile doldururuz.



num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
num_tokens.shape

# Bütün text verileri içinde maximum token sayısına sahip olanı buluyoruz
max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens) 
max_tokens = int(max_tokens)
max_tokens

# Bütün verilere padding yapıyoruz ve bütün veriler aynı boyutta oluyor
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

# size
print(x_train_pad.shape)
print(x_test_pad.shape)

model = Sequential() # Kullanacağımız Keras modelini tanımlıyoruz

embedding_size = 50 # Her kelime için vektör boyutunu 50 olarak belirledik

#Kerasta bir embedding layer oluşturuyoruz ve rastgele vektörler oluşturuyoruz


# Modele embedding layer ekliyoruz
# embedding matris size = num_words * embedding_size -> 10.000 * 50
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer'))

# 3-katmanlı(layer) LSTM
model.add(LSTM(units=16, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=8, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=4, return_sequences=False))
model.add(Dropout(0.2))
# Dense layer: Tek nörondan oluşuyor
model.add(Dense(1, activation='sigmoid'))# Sigmoid aktivasyon fonksiyonu

# Adam optimizer
from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr=1e-3)

# Farklı optimizerları deniyoruz
model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])

# modelin özeti
model.summary()

# epoch -> veri ile kaç kere eğiteceğiz
# batch_size -> feeding size-her epochta kaç veri ile besleyeceğiz
model.fit(x_train_pad, y_train, epochs=10, batch_size=256)

# model sonuçları
result = model.evaluate(x_test_pad, y_test)
result

# doğruluk oranı
accuracy = (result[1]) * 100
accuracy

#test yorumları(inputlar)
text1 = "böyle bir şeyi kabul edemem"
text2 = "tasarımı güzel ancak ürün açılmış tavsiye etmem"
text3 = "bu işten çok sıkıldım artık"
text4 = "kötü yorumlar gözümü korkutmuştu ancak hiçbir sorun yaşamadım teşekkürler"
text5 = "yaptığın işleri hiç beğenmiyorum"
text6 = "tam bir fiyat performans ürünü beğendim"
text7 = "Bu ürünü beğenmedim"
texts = [text1, text2,text3,text4,text5,text6,text7]

tokens = turkish_tokenizer.texts_to_sequences(texts)

tokens = turkish_tokenizer.texts_to_sequences(texts)
tokens

#padding
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)

#model bu yorumların hangi duyguya yakın olduğunu tahminliyor
model.predict(tokens_pad)

for i in model.predict(tokens_pad):
    if i < 0.5:
        print("negatif")#negatif yorum yapmış
    else
        print("pozitif")#pozitif yorum yapmış

from keras.models import load_model

model.save('hack_model.h5')  # modeli kaydediyoruz
