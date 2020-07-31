# Takım Üyeleri
> - Ümit Yılmaz [@umitylmz](https://github.com/umitylmz)
> - Büşra Gökmen [@newsteps8](https://github.com/newsteps8)
# FIXY

Amacımız Türkçe NLP literatüründeki birçok farklı sorunu bir arada çözebilen, eşsiz yaklaşımlar öne süren ve literatürdeki çalışmaların eksiklerini gideren open source bir yazım destekleyicisi/denetleyicisi oluşturmak. Kullanıcıların yazdıkları metinlerdeki yazım yanlışlarını derin öğrenme yaklaşımıyla çözüp aynı zamanda metinlerde anlamsal analizi de gerçekleştirerek bu bağlamda ortaya çıkan yanlışları da fark edip düzeltebilmek.

# Anlamsal yazım yanlışı düzeltme

Literatürde her ne kadar yazım yanlışlarını düzelten birçok kütüphane olsa da bunların hiçbiri anlamsal bağlamda meydana gelen hataları düzeltme kapasitesine sahip değildi. Bu tarz hataların en önemlilerinden olan bağlaç ve ek olan durumlarda ayrı veya bitişik yazılması gereken -da/-de, -ki, ve -mi örnek olarak verilebilir. Çalışmamız tamamıyla eşsiz olup, literatürdeki bütün örneklerinden çok daha iyi bir performans sergilemiştir. Model olarak 1DCNN, GRU, LSTM RNN denenmiş olup en iyi performans veren model olarak 2 layerlı Bidirectional LSTM seçilmiştir, bayesian search optimization tekniği ile parametreleri optimal değerlere ulaştırılmıştır.

Modelin performansının hem resmi hem de resmi olmayan dilde güzel sonuç vermesi ve genelleşmesi adına 3 farklı dataset birleştirilmiştir. Bunlar OPUS Subtitle veriseti, TSCORPUS Wikipedia veriseti ve TSCORPUS gazete verisetidir.

Çalışmamızda 85 milyondan fazla satırda veri işlenmiştir. Ön işleme olarak bitişik olan ekler kelimelerden ayrılmış ve “X” ile yer değiştirilmiştir. Ayrı olan ekler de “X” ile yer değiştirilmiştir. Bu sayede modelin aynı cümlede birden fazla ek olması durumunda da daha doğru sonuçlar vermesi amaçlanmıştır. Ön işleme işleminin ardından ayrı olan ekleri içeren cümleler 0, bitişik olan ekleri içeren cümleler ise 1 ile etiketlenmiştir. Ardından modelin hatalı öğrenmesine karşı 0 ve 1 labellarından fazla sayıda olanlar random undersampling ile azaltılmıştır. Sonuç olarak model hataları değil eklerin nasıl doğru yazılması gerektiğini öğrenmiştir. Oluşturulan verisetlerinin %20’si test %10’u da validation verisi olarak kullanılmıştır. Tüm verisetlerine csv formatında data klasörünün içerisinden ya da aşağıdaki başlıklar altında bulunan drive linklerinden ulaşabilirsiniz.

Modelleri pre_trained_weights klasöründe bulunan weightler ile doğrudan yükleyip test edebilirsiniz. Ya da siz de kendi düzelticinizi kendi verilerinizle veya bizim paylaştığımız veriler ve kendi oluşturduğunuz modeller ile train notebooklarını kullanarak eğitebilirsiniz.

Performans sonuçlarını, karşılaştırmalarını, modeller tarafından ayırt edilen cümle örneklerini ve verisetlerinin drive linklerini aşağıdaki başlıklar altında ayrı ayrı bulabilirsiniz.

## DE/-DA İÇİN LİTERATÜR KARŞILAŞTIRMASI

| Yapılan Çalışmalar | Doğruluk Oranı |F1 Score|
| ------ | ------ | ------ |
| Fixy |%92.13|%92.23|
| Boğaziçi | %76.48 |%86.67|
| Google Docs | %34 |%--|
| Microsoft Office  |%29|%--|
| ITU | %0 |%--|
| Libra Office | %0 |%--|

Kullanılan metodoloji tamamiyle özgündür ve Literatürdeki diğer çalışmalardan farklı bir yaklaşıma dayanmaktadır. Performans sonuçları yaklaşımın doğruluğunu ispatlar niteliktedir.

# DE Düzeltici

> - Accuracy on Test Data: 92.13%
> - ROC AUC on Test Data: 0.921

Confusion Matrix
[336706  20522]
[ 36227 327591]

| class | precision | recall | f1-score  |support
| ------ | ------ | ------ | ------ |------ |
| 0 | 0.9323 | 0.8912 | 0.9113 |30424
| 1 |  0.8958 |  0.9353 |0.9151|30425

Data
Oluşturulan 304244 satır veri içeren etiketli -ki veriseti linki: 
[Data](https://drive.google.com/file/d/1HLA9z1QoLMQsni70riq8APj0Gp_2nMmg/view?usp=sharing)

# Kİ Düzeltici

> - Accuracy on Test Data: 91.32%
> - ROC AUC on Test Data: 0.913

Confusion Matrix
 [27113  3311]
 [ 1968 28457]

| class | precision | recall | f1-score  |support
| ------ | ------ | ------ | ------ |------ |
| 0 | 0.9323 | 0.8912 | 0.9113 |30424
| 1 |  0.8958 |  0.9353 |0.9151|30425

Oluşturulan 304244 satır veri içeren etiketli -ki veriseti linki: 
[Data](https://drive.google.com/file/d/1HLA9z1QoLMQsni70riq8APj0Gp_2nMmg/view?usp=sharing)

# Mİ Düzeltici

Oluşturulan 9507636 satır veri içeren etiketli -mi veriseti linki: 
[Data](https://drive.google.com/file/d/1vCPsqYSMLOFxCA1WeykVMx1fT-A8etlD/view?usp=sharing)


> - Accuracy on Test Data: 95.41%
> - ROC AUC on Test Data: 0.954

Confusion Matrix
[910361  40403]
[ 46972 903792]

| class | precision | recall | f1-score  |support
| ------ | ------ | ------ | ------ |------ |
| 0 | 0.9509 | 0.9575 |0.9542  | 950764
| 1 |  0.9572 |  0.9506 | 0.9539|950764

Literatürde ki ve mi ekleri üzerine yapılmış çalışmaya rastlanamaması projenin özgünlüğünü arttırmaktadır. 

# Anlamsal Metin Analizi

### Türkçe Sentiment Analizi

Üç katmanlı LSTM nöral ağıyla oluşturduğumuz modeli yaklaşık 260000 tane pozitif ve negatif olarak etiketlenmiş sentiment verisiyle eğittik. Rastgele oluşturduğumuz kelime vektörleri ile birlikte embedding layeri nöral ağımıza ekledik. 10 Epoch ile eğittiğimiz modelden %94.57 accuracy(doğruluk) skoru elde ettik.

# Gerekli Kütüphaneler

```py
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,  Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
```
Kütüphaneleri yükledikten sonra keras ile modeli yüklüyoruz.

```py

from keras.models import load_model

model = load_model('hack_model.h5')  # modeli yüklüyoruz

```
Test inputları oluşturuyoruz.

```py
#test yorumları(inputlar)
text1 = "böyle bir şeyi kabul edemem"
text2 = "tasarımı güzel ancak ürün açılmış tavsiye etmem"
text3 = "bu işten çok sıkıldım artık"
text4 = "kötü yorumlar gözümü korkutmuştu ancak hiçbir sorun yaşamadım teşekkürler"
text5 = "yaptığın işleri hiç beğenmiyorum"
text6 = "tam bir fiyat performans ürünü beğendim"
text7 = "Bu ürünü beğenmedim"
texts = [text1, text2,text3,text4,text5,text6,text7]
```
Test inputları için tokenize ve padding yapıyoruz
```py
#tokenize
tokens = turkish_tokenizer.texts_to_sequences(texts)
#padding
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
```
model bu inputların hangi duyguya yakın olduğunu tahminliyor
```py
for i in model.predict(tokens_pad):
    if i < 0.5:
        print("negatif")#negatif yorum yapmış
    else
        print("pozitif")#pozitif yorum yapmış
```
```py
negative
negative
negative
positive
negative
positive
positive
```

Oluşturulan 260000 satır veri içeren etiketli pozitif-negatif veriseti linki:
[Data](https://drive.google.com/file/d/1--Az8CqFp4OljKHEbksim0ifIlFtAXDU/view?usp=sharing)

### Türkçe Formal-Informal Metin Analizi

Üç katmanlı LSTM nöral ağıyla oluşturduğumuz modeli twitter, newspaper ve wikipediadan aldığımız verinin yaklaşık 2504900 tanesini formal(düzgün) ve informal(düzgün olmayan) olarak etiketledik ve nöral ağımızı eğittik. Rastgele oluşturduğumuz kelime vektörleri ile birlikte embedding layeri nöral ağımıza ekledik. 10 Epoch ile eğittiğimiz modelden % 95.37 accuracy(doğruluk) skoru elde ettik.

# Gerekli Kütüphaneler

```py
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,  Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
```
Kütüphaneleri yükledikten sonra keras ile modeli yüklüyoruz.

```py

from keras.models import load_model

model = load_model('MODEL_FORMAL.h5')  # modeli yüklüyoruz

```
Test inputları oluşturuyoruz.

```py
# test inputları oluşturuyoruz
text1 = "atatürk, bu görevi en uzun süre yürüten kişi olmuştur."
text2="bdjfhdjfhdjkhj"
text3 = "hiç resimde gösterildiği gibi değil..."
text4 = "bir yirminci yüzyıl popüler kültür ikonu haline gelen ressam, resimlerinin yanı sıra inişli çıkışlı özel yaşamı ve politik görüşleri ile tanınır. "
text5 = "fransız halkı önceki döneme göre büyük bir evrim geçirmektedir. halk bilinçlenmektedir ve sarayın, kralın, seçkinlerin denetiminden çıkmaya başlamıştır. şehirlerde yaşayan pek çok burjuva, büyük bir atılım içindedir. kitaplar yaygınlaşmakta, aileler çocuklarını üniversitelere göndererek sağlam bir gelecek kurma yolunu tutarak kültürel seviyeyi yükseltmektedir. bağımsız yayıncıların çıkardıkları gazete, bildiri ve broşürler, kitlesel bilinçlenmeye yol açmaktadır. bu koşullar da toplumsal değişim taleplerinin olgunlaşmasına yol açmıştır."
text6 = "bunu çıkardım söylediklerinden"
text7 = "Bu koşullar da toplumsal değişim taleplerinin olgunlaşmasına yol açmıştır."
text8="bu çok saçma yaa"
text9="bana böyle bir yetki verilmedi."
text10="napıcaz bu işi böyle"
text11="Öncelikle Mercedes-Benz’e olan ilgin için teşekkür ederiz."
text12="Ekibimizle çalışma isteğin için teşekkür ediyor, sağlıklı günler ve kariyerinde başarılar diliyoruz. Farklı etkinlik ve programlarda tekrar bir araya gelmek dileğiyle."
text13="Ben de öyle olduğunu düşünmüyordum ama gittik yine de jzns"
texts = [text1, text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13]
```
Test inputları için tokenize ve padding yapıyoruz
```py
#tokenize
tokens = tokenizer.texts_to_sequences(texts)
#padding
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
```
model bu inputların hangi duyguya yakın olduğunu tahminliyor
```py
#test verisini tahminleme
for i in model.predict(tokens_pad):
  if i < 0.5:
    print("informal")
  else:
    print("formal")
```
```py
formal
informal
informal
formal
formal
informal
formal
informal
informal
informal
formal
informal
informal
```
Oluşturulan 1204900 satır veri içeren etiketli formal veriseti linki:
[Data](https://drive.google.com/file/d/1-8UDP-WYRRXEcbr1HxBwuXSp1GW8Ta5-/view?usp=sharing)
Oluşturulan 3934628 satır veri içeren etiketli informal veriseti linki:
[Data](https://drive.google.com/file/d/1UBxgxLPv_afebNgm9sFt2n2WjMwgc_YK/view?usp=sharing)


### Türkçe Emotion(Duygu) Metin Analizi
 6  farklı duygu(Fear: Korku,Happy: Sevinç,Sadness: Üzüntü,Disgust: İğrenme-Bıkkınlık,Anger: Öfke,Suprise: Şaşkınlık) ile etiketli 27350 verinin bulunduğu veriseti ile SVM linearSVC,MultinomialNB,LogisticRegression, RandomForestClassifier modellerini eğittik. Modellemeden önce verideki kelimeleri vektörize etmek için tfidf vektörizerı ve türkçe stopword listesini kullandık. Bu modeller arasında en yükse accuracy oranını LinearSVC modeli ile elde ettik.
 
 
# Model Accuracy Oranlarının Karşılaştırılması

|Model|Accuracy(Doğruluk Oranı)|
|-----|-----|
|LinearSVC |0.80|
|LogisticRegression |0.79|
|MultinomialNB|0.78|
|RandomForestClassifier|0.60|


# Gerekli Kütüphaneler

```py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
Kütüphaneleri yükledikten sonra load_model ile modeli yüklüyoruz ve test ediyoruz.

```py

# modeli yükleyip test ediyoruz
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=myList)

loaded_model = pickle.load(open("emotion_model.pickle", 'rb'))
corpus = [
         "İşlerin ters gitmesinden endişe ediyorum",
         "çok mutluyum",
         "sana çok kızgınım",
         "beni şaşırttın",
        ]
tfidf.fit_transform(df.Entry).toarray()
features = tfidf.transform(corpus).toarray()
result = loaded_model.predict(features)
print(result)

```
```py
['Fear' 'Happy' 'Anger' 'Suprise']
```
Bu model için kullandığımız verisetini [TREMODATA](http://demir.cs.deu.edu.tr/tremo-dataset/) adresinden
istedik. Ayrıca bu verisetinin bulunduğu drive linki: [Data](https://drive.google.com/file/d/1t0Ffu1edduOi8HNfGDUPcq6H5Iqfi4GF/view?usp=sharing)

# fixy_app(Flask API)

Virtual environment ortamında oluşturduğunuz Flask API yardımı ile modelleri bir arayüzde gösterebilirsiniz. Bunun için gerekli kütüphaneler:

```py
from flask_wtf import FlaskForm
from flask import Flask, request, render_template,redirect
import pickle
import re
from wtforms.validators import DataRequired
import pandas as pd
from os.path import join
```

app.py dosyasını çalıştırarak modellerinizi ön yüze bağlayabilirsiniz.

![alt text](https://github.com/umitylmz/fixy/blob/master/fixy_app_1.jpeg)

![alt text](https://github.com/umitylmz/fixy/blob/master/fixy_app_2.jpeg)


#### Temizlenmiş Wikipedi Veriseti

2364897 satırlık temizlenmiş Türkçe Wikipedia verisetini herhangi bir Türkçe NLP çalışmasında kullanabilirsiniz:)
[Wikipedia Veriseti](https://drive.google.com/file/d/1ujQMxIUEXuIOihU4-cRhGwMrR3YVoYi3/view?usp=sharing)

