# fixy

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

> Accuracy on Test Data: 92.13%
> ROC AUC on Test Data: 0.921

Confusion Matrix
[336706  20522]
[ 36227 327591]

| class | precision | recall | f1-score  |support
| ------ | ------ | ------ | ------ |------ |
| 0 | 0.9323 | 0.8912 | 0.9113 |30424
| 1 |  0.8958 |  0.9353 |0.9151|30425


Oluşturulan 304244 satır veri içeren etiketli -ki veriseti linki: 
[https://drive.google.com/file/d/1HLA9z1QoLMQsni70riq8APj0Gp_2nMmg/view?usp=sharing]

# Kİ Düzeltici

> Accuracy on Test Data: 91.32%
> ROC AUC on Test Data: 0.913

Confusion Matrix
 [27113  3311]
 [ 1968 28457]

| class | precision | recall | f1-score  |support
| ------ | ------ | ------ | ------ |------ |
| 0 | 0.9323 | 0.8912 | 0.9113 |30424
| 1 |  0.8958 |  0.9353 |0.9151|30425

Oluşturulan 304244 satır veri içeren etiketli -ki veriseti linki: 
[https://drive.google.com/file/d/1HLA9z1QoLMQsni70riq8APj0Gp_2nMmg/view?usp=sharing

# Mİ Düzeltici

Oluşturulan 9507636 satır veri içeren etiketli -mi veriseti linki: 
[https://drive.google.com/file/d/1vCPsqYSMLOFxCA1WeykVMx1fT-A8etlD/view?usp=sharing]


> Accuracy on Test Data: 95.41%
> ROC AUC on Test Data: 0.954

Confusion Matrix
[910361  40403]
[ 46972 903792]

| class | precision | recall | f1-score  |support
| ------ | ------ | ------ | ------ |------ |
| 0 | 0.9509 | 0.9575 |0.9542  | 950764
| 1 |  0.9572 |  0.9506 | 0.9539|950764

Literatürde ki ve mi ekleri üzerine yapılmış çalışmaya rastlanamaması projenin özgünlüğünü arttırmaktadır. 

# Anlamsal Metin Analizi





