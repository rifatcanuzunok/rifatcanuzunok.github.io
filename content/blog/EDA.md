---
author: "Rıfatcan Uzunok"
date: 2022-02-24
title: Keşifçi Veri Analizi
---

## Keşifçi Veri Analizi (/EDA) Nedir?

Keşifçi veri analizi, elimizdeki veri setinin ana bileşenlerinin özetini çıkararak ve genelde grafikler kullanarak anlam çıkarmaya yarayan bir analiz yöntemidir. Özellikle bir makine öğrenmesi modeli kurmadan önce bu aşama kritik derecede önemlidir. Hem elimizde veri seti hakkında bize önemli fikirler verip, veriyi daha iyi tanımamızı sağlar hem de makine öğrenmesi algoritmalarına veri setini düzgün bir şekilde verdiğimizden emin olmamızı sağlar. Histogram, kutu grafiği ve saçılım grafiği sıklıkla başvurulan grafik yöntemlerdendir.

## Keşifçi Veri Analizi Nasıl Yapılır?

Keşifçi veri analizi için kesin olarak uygulanması gereken yöntemler ve takip edilmesi gereken adımlar yoktur. Bu yüzden bu sorunun cevabı elimizdeki veri setine göre değişkenlik gösterecektir. Ancak bu yazıda bir örnek üzerinden keşifçi veri analizi nasıl yapılır, genel bir fikir vermesi açısından göstermeye çalışacağım.

## Veri Kaynağı

Analiz edeceğimiz veriyi bulmak için temelde 2 kaynağımız vardır.

1- Özel Kaynaklar

2- Açık Kaynaklar

### Özel Kaynaklar

İsminden de anlaşılabileceği gibi, özel veri kaynakları genellikle özel şirketler tarafından, sağlanır ve genellikle şirketle ilgili iç analizler yapılması için sağlanır. Bu tarz verilere genellikle veri sağlayıcısıyla bir bağlantımız yok ise ulaşamayız.

### Açık Kaynaklar

Herkesin ulaşabileceği veri kaynaklarıdır. Herhangi ekstra bir izne ihtiyaç olmaksızın herkesin alıp kullanabileceği verileri içerisinde barındırırlar.

- [https://www.kaggle.com/](https://www.kaggle.com/)
- [https://archive.ics.uci.edu/ml/index.php](https://archive.ics.uci.edu/ml/index.php)
- [https://github.com/awesom/EDAta/awesome-public-datasets](https://github.com/awesom/EDAta/awesome-public-datasets)

Keşifçi Veri Analizine başlayabilmemiz için öncelikle veri setini bulmamız gerekiyordu. Açık veri seti kaynaklarına birkaç örnek verdiğimize göre, Veri temizleme adımına geçebiliriz.

## Veri Temizleme

Yeni bir veri seti ile karşılaştığımızda ondan belli anlamlar çıkarabilmek için öncelikle veri setinin temel hatlarına hakim olduğumuzdan emin olmamız gerekir. Veri temizleme adımı temelde veri içerisindeki düzensizlikleri temizleyip, veriye genel bir çerçeve çizdiğimiz adım olarak karşımıza çıkar. Veri içerisindeki düzensizlikler karşımıza

- Eksik Değerler
- Yanlış tanımlanmış değişken tipleri
- Yanlış değişken isimleri
- Yanlış değerler
- Uç değerler / Anomaliler

gibi pek çok farklı şekilde çıkabilir.

Bu yazıda kullandığım veri seti [buradan](https://github.com/Kaushik-Varma/Marketing_Data_Analysis) bulunabilir.

```python
# Kullanacağımız kütüphaneleri import ediyoruz.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Marketing Analysis isimli veri setini okutuyoruz.
df = pd.read_csv('datasets/Marketing_Analysis.csv')
# İlk 5 gözleme bakalım
df.head()
```

![Untitled](/EDA/Untitled.png)

Veri setine baktığımızda, doğru olmayan bazı şeyler olduğu net bir şekilde gözümüze çarpıyor. İlk satırda veri setinin başlığı, diğer satırda ise veri setlerinin toplu isimleri görülüyor. Ancak veri yapımıza uygun olması için değişken isimlerinin bulunduğu satırdan itibaren okunması gerekirdi. Bu yüzden ilk 2 satırı atlayıp veriyi tekrar okutuyoruz.

```python
df = pd.read_csv('datasets/Marketing_Analysis.csv', skiprows=2)
df.head()
```

![Untitled](/EDA/Untitled1.png)

Bu şekilde baktığımızda çok daha anlamlı bir veri seti yapısı elde ettiğimizi görebiliyoruz.

Genel olarak bir veri setinin satır ve sütunlarıyla ilgili yapmamız gereken düzenlemeler şu şekilde sıralanabilir.

1. Satır ve sütunlarda özet bilgi içeren değerler silinir. (Örnek veri setindeki ilk 2 satır gibi.)
2. Başlık ve altbilgilerin gözlem birimleri içerisinde göründüğü durumlarda silme işlemi yapılır.
3. Tamamen boş olan gözlemler silinir.
4. Birkaç değişken, daha anlamlı tek bir değişken haline getirilebilir.
5. Birkaç değişkenlik bilgiyi içeren bir değişken, ayrı değişkenler haline getirilebilir.
6. Değişken isimleri düzgün şekilde yazılmadıysa, bunlara isim verilebilir.

Veri setimize bakarsak, `customerid` değişkeninin herhangi bir bilgi içermediğini, bu yüzden onu atmanın bir sorun yaratmayacağını anlayabiliriz. Aynı şekilde `jobedu` değişkeni içerisinde gözlem birimlerimiz hakkında 2 ayrı ve önemli bilginin taşındığı görülebilir. `jobedu` değişkeni içerisinden `job` ve `education` gibi yeni değişkenler üretmek mümkün.   

```python
# customerid değişkenini veri setinden atalım.
df.drop('customerid', axis=1, inplace=True)
# jobedu değişkeni içerisindeki iş ve eğitim bilgilerini ayrı değişkenler olarak atayalım.
df['job'] = df['jobedu'].apply(lambda x: x.split(',')[0])
df['education'] = df['jobedu'].apply(lambda x: x.split(',')[1])
# içerisindeki bilgiyi aldığımız jobedu değişkenine artık ihtiyacımız yok,
# bu değişkeni de atabiliriz.
df.drop('jobedu', axis=1, inplace=True)
```

![Untitled](/EDA/Untitled2.png)

### Eksik Değerler

Veri setimiz ile herhangi istatistiki bir analiz yapmadan önce kesinlikle göz önünde bulundurmamız gereken bir diğer durum ise eksik değerlerdir. Verimiz içerisinde eksik değerler temelde 2 sebep ile bulunabilir.

1. Tamamen Rastgele Oluşan Eksik Değerler:

    Bu tarz veriler, veri seti içerisinde herhangi bir değişkene bağlı olmayan ve rastgele olarak oluşmuş eksik verilerdir.

2. Rastgele olmayan eksik değerler:

    Bu tarz eksik değerler, diğer bazı değişkenlerle bulunduğu bağlantıdan dolayı ya da eksik olmasının da başka bir anlam taşıması sebebi ile eksiktir.


Veri setindeki eksik değerlere bir bakalım

```python
df.isnull().sum()
```

![Untitled](/EDA/Untitled3.png)

Gördüğümüz gibi, veri seti içerisindeki 3 değişkende eksik değerlerimiz mevcut. Eksik değerlerle başa çıkabilmek için izleyebileceğimiz çeşitli yöntemler var ancak bu yöntemlerden hangisini tercih edeceğimiz elimizdeki veri setine göre değişecektir.

1. Eksik Değerleri Silmek:

    Bu yöntem ilk akla gelen ve en basit yöntem diyebiliriz. Eksik değer bulunduran gözlem birimlerini direkt olarak veri setinden çıkarıp, analizlere bu şekilde devam etmeyi hedefler. Eğer elimizdeki veri seti *yeterince büyük* ise bu yöntem rahatlıkla kullanılabilir.

    Artıları:

    - Eksik değerlere sahip verilerin tamamen atılması, ileride kurulacak modeli sağlam ve doğru yapacaktır.
    - Önemli bir bilgi içermeyen satırları silmiş olmak, veri setinde değersiz bir kalabalık olmamasını sağlayacaktır.

    Eksikleri:

    - Gözlem ve bilgi kaybı
    - Eksik Değer oranı fazla ise bulduğumuz sonuçlar gerçekçi olmaktan uzaklaşacaktır. Örneğin veri seti içerisindeki gözlemlerin %30unda boş değerler varsa ve bunları tamamen silersek, veri seti içerisindeki bilinin %30luk bir kısmını kaybettiğimiz için sonuçlarımız gerçek dünya ile uyumlu olmayabilir.

    ```python
    # Boş olan gözlemleri veri setinin dışında bırakıyoruz.
    df = df[~df.age.isnull()]
    ```

2. Eksik değerleri başka değerler ile doldurmak:

    Sayısal değişkenler için kullanabileceğimiz bu yöntem, eksik olan verileri o değişkenin ortalaması, medyanı ya da mod değeri ile doldurmayı hedefler. Böylece eksik olan gözlem birimini atarak bilgi kaybetmek yerine, veri setinin yapısını bozmayacak bir değer ile doldurup analizlere devam edebilmeyi hedefler.

    Artıları:

    - Veri seti küçük olduğunda daha iyi sonuçlar alındığı gözlenebilir.
    - Veri kaybı yaşamayı önler.

    Eksileri:

    - Tüm eksik değerleri aynı değer ile doldurmak veri setine belli bir yanlılık ve varyans kazandırır.
    - Tek bir değer ataması yapmak, diğer atama yöntemlerine göre daha kötü sonuçlar verir.

    ```python
    # 'month' değişkeninin modunu hesaplayıp, boş değerler yerine bu değeri veriyoruz.
    month_mode = df['month'].mode()[0]
    df['month'].fillna(month_mode, inplace=True)
    ```

3. Eşsiz bir kategori ataması yapmak:

    Kategorik değişkenlerin belli bir sayıda eşsiz değeri olabilir ve bunlara kategori ismi verilir. Örneğin cinsiyet değişkeninin alabileceği yalnızca iki farklı değişken vardır. Buradaki eksik değerlerimizi farklı bir kategori olarak adlandırıp işlemlerimize devam edebiliriz.

    Artıları:

    - Yeni bir kategori ekleyerek veri kaybını önler.
    - Tüm eksik değerleri tek bir kategori olarak gördüğü için *one hot encoding* yapıldıktan sonra daha düşük varyans ile karşılaşılır.

    Eksileri:

    - Modele yeni bir değişken eklediği için kötü performans verebilir.

    ```python
    # yalnızca iki kategorisi olan 'response' değişkeninde
    # bilinmeyen değerlerin tamamı U olarak atandı.
    df.response.fillna('U', inplace=True)
    ```

4. Eksik Değerleri Tahmin Etmek:

    Eksik değeri olmayan değişkenleri kullanarak eksik değerleri tahmin edebiliriz. Bu yöntem eğer eksik değerlerde büyük varyans olmasını beklemiyorsak iyi sonuçlar verebilir. Örnek olarak doğrusal regresyon ile yaş değişkenindeki boş değerleri tahmin edebiliriz. Daha isabetli bir karar vermek istiyorsak tek bir algoritma yerine birden çok algoritma kullanıp, aralarından en isabetli olduğunu düşündüğümüz seçilebilir.

    Artıları:

    - Eksik olan değerleri tahmin etmiş olmak model için bir gelişme sayılabilir.
    - Model parametrelerinin yansız tahmincilerini bulmuş oluruz.

    Eksileri:

    - Diğer değişkenler içerisinde eksik, yanlış gözlemler bulunması tahmin etmeye çalıştığımız gözlemleri de etkileyeceğinden yanlılık artabilir.
    - Atanan değerlere gerçek değerlerin vekilleri olarak bakılmalı.

    ```python
    # Veri seti içerisindeki sayısal değişkenleri seçiyoruz.
    num_cols = [col for col in df.columns if df[col].dtypes != 'O']

    from sklearn.linear_model import LinearRegression
    df_with_null = df.copy()
    df_without_null = df_with_null.dropna()

    # Age haricinde tüm değişkenler
    X_train = df_without_null[num_cols].drop('age', axis=1)
    # Yalnızca Age değişkeni
    y_train = df_without_null['age']

    # modeli fit ediyoruz.
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    test_data = df_with_null[num_cols].drop('age', axis=1)
    predicted_age = pd.DataFrame(lr.predict(test_data))
    df[df['age'].isna()].index
    predicted_age.loc[df[df['age'].isna()].index]
    ```

5. Kategorik Değişken Kırılımı ile Atama Yapmak:

    Bu yöntem eksik değerleri ortalama, mod, medyan ile doldurma temeline dayanmakla birlikte, başka bir kategorik değişkenin sınıflarının ortalama, mod ya da medyanını kullanır. Örnek olarak veri setimizdeki gözlemlerin `job` değişkenine göre kırılımını alıp yaş ortalamalarına bakabiliriz. Bu şekilde meslek gruplarına göre yaş ortalamalarını bulmuş oluruz. Bu durum, boş olan tüm gözlemlere aynı değeri vermek yerine, daha isabetli bir şekilde atama yapmamızı sağlar.



    ```python
    df.groupby("job")["age"].mean() # job değişkenine göre yaş ortalamalarını alıyoruz.
    # boş olan gözlemlere ilgili mesleğe ait yaş ortalamasını atıyoruz.
    df['age'].fillna(df.groupby('job')['age'].transform('mean'))
    ```

6.  KNN Imputer ile eksik değer doldurmak:

    KNN uzaklık temelli bir makine öğrenmesi algoritmasıdır. Boş gözlem değerlerini doldurmak için de kullanılabilen bu yöntemde


## Aykırı Değerler

Diğer gözlemlerden *kayda değer derecede* uzak olan gözlemlere aykırı veya uç değer denir. Aykırı değer problemi, değişkenin kendi içerisindeki değerlerden kaynaklanıyor olabilir. Buna Univariate Outliers(Tek değişkenli aykırı değer) denir. Aynı zamanda başka bir değişken ile olan ilişki incelenirken bu iki değişkenin hareketi birlikte incelendiğinde aykırı olan bazı gözlemlere rastlanabilir. Bunlara Multivariate Outliers (Çok değişkenli aykırı değer) denir.

![Untitled](/EDA/Untitled4.png)

Aykırı değerlerin sebebini anladıktan sonra, bu değerleri veri setimizden atabilir, yerine başka değerler atayabilir ya da hiçbir şey yapmamayı tercih edebiliriz.

1. Aykırı değerlere ulaşmak:
    1. *Grafik Yöntem:*

        ```python
        sns.boxplot(x=df['age'])
        plt.show()
        ```

        ![Untitled](/EDA/Untitled5.png)

        Boxplot yardımı ile aykırı değerlerin varlığı hakkında hızlı bir fikir sahibi olabiliriz. Kutunun sağ ve solundaki çizgiler verimizin 1. ve 3. kartillerini, ortadaki çizgi ise medyan değerini ifade ediyor. Kutunun sağ ve sol uçlarındaki çizgiler ise alt ve üst sınırlarımızı ifade ediyor. Bu çizgilerin dışarısında kalan ve nokta olarak gösterilen değerler de aykırı değer olarak adlandırılır.

    2. *Aralık hesabı ile ulaşma:*

        ```python
        # 1. ve 3. kantiller hesaplanır
        q1 = df["age"].quantile(0.25)
        q3 = df["age"].quantile(0.75)
        # bunların farkı bize kartiller arası mesafeyi verir.
        iqr = q3 - q1
        # üst sınıra 1.5 iqr ekleyip, alt sınırdan 1.5 iqr çıkararak alt ve üst sınır
        # bulunmuş olur.
        up = q3 + 1.5 * iqr
        low = q1 - 1.5 * iqr
        # alt sınırdan küçük ya da üst sınırdan büyük olanlar
        df[(df["age"] < low) | (df["age"] > up)]
        # aykırı değere sahip gözlem birimlerinin indexleri
        df[(df["age"] < low) | (df["age"] > up)].index
        ```


    2. Aykırı değer problemini çözmek:

    1. *Silme:*

        ```python
        df[~((df["age"] < low) | (df["age"] > up))]
        ```

        ~ yaptığımız sorgu sonucunda çıkan değerin/değerlerin değilini alan bir operatördür. Bu örnek için, yaşı alt sınırdan küçük veya üst sınırdan büyük olanlar parantez içerisinde bulunur. Daha sonrasında ~ operatörü nedeniyle olmayanlar seçilir. Yani sonuç olarak alt sınırdan büyük ve üst sınırdan küçük değerleri bize döndürür. Böylece aykırı değerleri veri setimiz dışarısında bırakmış oluruz.

    2. *Baskılama:*

        Gözlemler içerisinde alt sınırdan küçük ve üst sınırdan büyük olan değerleri bu sınırlara eşitleyecek şekilde baskılama işlemi yapılabilir.

        ```python
        # alt sınırdan küçük olan değerleri alt sınır ile değiştiriyoruz.
        df.loc[(df['age'] < low), 'age'] = low
        # üst sınırdan büyük olan değerleri üst sınır ile değiştiriyoruz.
        df.loc[(df['age'] > up), 'age'] = up
        ```

        ![Untitled](/EDA/Untitled6.png)

        Bu işlemlerden sonra boxplot'a tekrar baktığımızda herhangi bir aykırı gözlem kalmadığını görüyoruz. Baskılama ya da silme işlemlerinde bilgi kaybı oluştuğu için bu işlemler yapılırken bu durum mutlaka göz önünde bulundurulmalıdır.


    ## Çok Değişkenli Aykırı Değer Analizi

    ### Local Outlier Factor (LOF)

    LOF veri seti içerisindeki aykırı değerleri saptamaya yarayan bir gözetimsiz makine öğrenmesi algoritmasıdır. Her örnek değeri için bir yerel yoğunluk puanı oluşturulur ve bunlara ağırlık verilir. Tüm puanlar karşılaştırılarak, veri setinde düşük puana sahip olan gözlemlerin anomali/aykırı değer olarak tanımlanması sağlanır.

    > "The local outlier factor is based on a concept of a local density, where locality is given by  nearest neighbors, whose distance is used to estimate the density. By comparing the local density of an object to the local densities of its neighbors, one can identify regions of similar density, and points that have a substantially lower density than their neighbors. These are considered to be outliers."
    >

    ![Untitled](/EDA/Untitled7.png)

    ```python
    from sklearn.neighbors import LocalOutlierFactor
    # LOF modelini kuralım
    clf = LocalOutlierFactor(n_neighbors=20)
    # sayısal değişkenlerimiz için fit edelim.
    clf.fit_predict(df[num_cols])
    df_scores = clf.negative_outlier_factor_
    # LOF skorları sıralanmış numpy arrayine bakalım
    np.sort(df_scores)[0:30]
    # bu skorları grafik üzerinden incelemek istersek
    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 20], style='.-')
    plt.show()
    ```

    ![Untitled](/EDA/Untitled8.png)

    ![Untitled](/EDA/Untitled9.png)

    Aykırı olmayan değerlerin genellikle daha yüksek LOF puanına sahip olacağı söylenebilir. Bulduğumuz puanlar arasından bir noktayı sınır olarak belirleyip bunun altında kalan değerleri veri setimizden çıkarmamız gerekir.

    Hem grafikten, hem de puanlarımızın olduğu array üzerinden inceleme yapabiliriz. Grafik üzerinde *eğim*de ciddi bir değişikliğin olduğu bir nokta seçilmelidir. Bu noktalara karşılık gelen değerler puan arrayimizde görülebiliyor. Seçtiğimiz sınır ne kadar büyük olursa o kadar çok gözlem aykırı değer olarak tanımlanacağı ve veri setinden atılacağı için, puanlarda birkaç ciddi atlama yapan kritik noktalar belirlenmeli ve seçilmeli. Bu örnek için sınırımız 7 olarak seçilebilir.

    ```python
    th = np.sort(df_scores)[7]
    df[df_scores < th] # aykırı gözlemlerin kendisine ulaşma
    df[df_scores < th].shape # aykırı gözlemlerin sayısını görme
    df[df_scores < th].index # aykırı gözlemlerin index bilgisine ulaşma
    ```


## Tek Değişkenli Veri Analizi

Veri seti içerisindeki tek bir değişkene odaklanarak yaptığımız analizlere, tek değişkenli veri analizi denir.  Burası değişken türlerine bakmak için doğru bir nokta olabilir.

### Kategorik Değişkenler

Kategorik değişkenler, yalnızca belli kategorik değerleri içerisinde bulundurabilen değişkenlerdir. Örneğin sayısal bir değişken olan yaş değişkenini, çocuk, genç, yetişkin gibi 4 kategoriye ayırırsak artık gerçek yaş değerlerine değil düştükleri aralıklara göre kategorilere bakıyor oluruz. Bu kategoriler arasında hiyerarşik bir ilişki varsa ordinal, yoksa nominal kategorik değişken olarak tanımlanabilir.

1. Nominal Kategorik Değişkenler:

    Kategoriler arasında herhangi hiyerarşik bir yapı bulunmayan kategorik veriler için kullanılan isimdir. Örneğin cinsiyet, kan grubu, doğum yeri, medeni durum.

2. Ordinal Kategorik Değişkenler:

    Ordinal veriler de yine kategorik veri türündendir. Fakat değerleri arasında sıralı bir ilişki bulunmaktadır. “Daha fazla” ifadesi ile kullanılabilirler ancak nekadar daha fazla olduğunun ölçüsünü veremezler. Örneğin: Eğitim Düzeyi, Sosyo ekonomik ölçek skorları gibi. Ordinal kategorik veriler, nominal verilere göre daha fazla bilgi taşır.


### Sayısal Değişkenler

Üzerinde matematiksel işlemler uygulayabildiğimiz değişkenlere sayısal değişkenler denir. Sürekli ve kesikli olarak iki ayrı kategoride incelenebilirler.

1. Kesikli Değişken:

    Yalnızca belli bir aralıktaki tam sayı değerlerini alabilen değişkenlere kesikli değişken denir. Örneğin sahip olunan araba sayısı.

2. Sürekli Değişken:

    Sürekli değişkenler tüm değerleri alabilir. Örneğin ağırlık, boy.


### Nominal Değişken Analizi

Nominal kategorik değişkenin ne demek olduğuna baktık, veri seti içerisindeki `job` değişkenimiz buna güzel bir örnek olabilir. `job` değişkeni kategorik bir değişken olduğundan bar plot ile grafiğini görebiliriz.

```python
# job değişkeninin nasıl dağıldığını görmek için
df['job'].value_counts(normalize=True)
# Dağılımı bar plot şeklinde görmek için
df['job'].value_counts(normalize=True).plot.barh()
plt.show()
```

![Untitled](/EDA/Untitled10.png)

Yukarıdaki grafik sayesinde, veri seti içerisinde bulunan kişiler arasında en yaygın meslek grubunun mavi yaka olduğunu söyleyebiliriz.

### Ordinal Değişken Analizi

Veri seti içerisindeki `education` değişkenimizin ordinal kategorik bir değişken olduğu söylenebilir.

```python
# education değişkeninin nasıl dağıldığını görmek için
df['education'].value_counts(normalize=True)
# pie chart ile kategorilerin gösterimi
df['education'].value_counts().plot.pie()
plt.show()
```

![Untitled](/EDA/Untitled11.png)

Veri seti içerisindeki insanların yarısının lise mezunu olduğunu ve onları %30 civarında üniversite mezunlarının takip ettiğini görebiliriz.

## İki Değişkenli Veri Analizi

1. Sayısal - Sayısal Analizi:

    İki sayısal değişkenin aynı anda incelenmesi için kullanabileceğimiz bazı yöntemler:

    - Scatter Plot (Saçılım grafiği)
    - Pair Plot
    - Korelasyon matrisi
    1. Scatter Plot:

        Saçılım grafiği, iki değişkenin birlikte hareketlerini incelemek için kullanılan bir grafik yöntemdir. Bu grafikte iki değişken arasında doğrusal bir ilişki olup olmadığı rahatlıkla görülebilir.

        ```python
        plt.title('Salary vs Balance')
        plt.scatter(df.salary, df.balance)
        plt.show()
        ```

        ![Untitled](/EDA/Untitled12.png)

        ```python
        plt.title('Age vs Balance')
        plt.scatter(df.age, df.balance)
        plt.show()
        ```

        ![Untitled](/EDA/Untitled13.png)

    2. Pair Plot:

        Aynı 3 değişken için pair plot grafiklerini de çizdirelim. Pair plot köşegenlerinde histogram, köşegen dışında ise kesişen 2 değişkenin saçılım grafiklerini bize verir. Tek tek tüm saçılım graifklerine bakmaktansa veri içerisindeki tüm sayısal değişkenlerin saçılım ve histogramlarını tek bir grafikte görmemizi sağlar.

        ```python
        sns.pairplot(df, vars=['salary', 'balance', 'age'])
        plt.show()
        ```

        ![Untitled](/EDA/Untitled14.png)

    3. Korelasyon Matrisi:

        Değişkenler arasında doğrusal bir bağlantı olup olmadığını, varsa da ne kadar kuvvetli ya da zayıf olduğunu ve yönünü belirten korelasyon çok değişkenli veri analizinde sıklıkla başvurulan bir yöntemdir. Hem tablo olarak hem de ısı haritası olarak grafiğini çıkarmak mümkündür.

        ```python
        sns.heatmap(df.corr(), annot=True, cmap='Reds')
        plt.show()
        ```

        ![Untitled](/EDA/Untitled15.png)

        Korelasyon değerleri +1e yaklaştıkça güçlü pozitif doğrusal bağlantıyı, -1e yaklaştıkça da güçlü negatif doğrusal bağlantıyı işaret eder. 0'a yakın olduğunda ise değişkenler arasında doğrusal bir bağlantı bulunmadığı yorumu yapılabilir.

2. Sayısal - Kategorik Değişken Analizi:

    Kategorik ve sayısal değişkenleri aynı anda incelemek için genellikle kırılımlar kullanılır ve bu kırılımlar sonrasında sayısal değişkenin ortalama, medyan gibi değerlerine bakılır

    `salary` ve `response` değişkenlerimize bakalım

    `response` değerlerine göre ortalama `salary` nasıl dağılıyor öncelikle buna bakalım.

    ```python
    df.groupby('response')['salary'].mean()
    ```

    ![Untitled](/EDA/Untitled16.png)

    `response`kırılımında maaşlar arasında ciddi bir farklılık olmadığını gözlemliyoruz. Aynı şekilde medyan değerlerini de inceleyelim

    ```python
    df.groupby('response')['salary'].median()
    ```

    ![Untitled](/EDA/Untitled17.png)

    Medyan değerleri arasında da anlamlı bir farklılık olmadığını gördük. Response değişkenine verilen evet ya da hayır cevaplarının maaş üzerinde bir değişiklik yaratmadığı görülüyor. Ancak gerçekten bir farklılık yaratıp yaratmadığını görmek için, bir de boxplot ile deneyelim.

    ```python
    sns.boxplot(df.response, df.salary)
    plt.show()
    ```

    ![Untitled](/EDA/Untitled18.png)

    Ortalama ve medyan değerlerine göre daha farklı bir sonuca ulaştığımız açıkça görülebilir. Evet olarak yanıt veren müşterilerin kutu grafiği incelendiğinde kutunun daha yukarıda olduğu görülebiliyor. Yani Evet cevabını veren müşterilerin maaşları yüksek olmaya daha yatkın gibi bir yorum buradan çıkarabiliriz. Ancak bu yorumun anlamlı olup olmadığı istatistiksel yöntemler kullanarak test edilmelidir.

3. Kategorik - Kategorik Değişken Analizi:

    Hedef değişkenimiz olan `response` içerisinde 'evet' ve 'hayır' olarak iki değer taşıyor. Bunları 1 ve 0 olarak kodlayıp analizimize bu şekilde devam edelim.

    ```python
    df['response_rate'] = np.where(df.response == 'yes', 1, 0)
    df.response_rate.value_counts()
    ```

    ![Untitled](/EDA/Untitled19.png)

    Şimdi de `response` değişkenimizin medeni hallere göre nasıl dağıldığına bakalım.

    ```python
    df.groupby('marital')['response_rate'].mean().plot.bar()
    plt.show()
    ```

    ![Untitled](/EDA/Untitled20.png)

    Yukarıdaki grafik sayesinde, bekar insanların aramaları daha fazla yanıtladığı çıkarımında bulunabiliriz.


### Çok Değişkenli Veri Analizi

İkiden fazla değişken ile yapılan analizlere çok değişkenli veri analizi denir.

`education`, `marital`, `response_rate` değişkenlerimiz birbirlerini nasıl etkiliyormuş bir bakalım.

Bunu yapmak için pivot table kullanabiliriz. Bu 3 değişkeni kullanarak bir pivot table elde edelim.

```python
# pivot table
result = pd.pivot_table(df, index='education', columns='marital', values='response_rate')
# heatmap
sns.heatmap(result, annot=True, cmap='RdYlGn', center=0.117)
```

![Untitled](/EDA/Untitled21.png)

![Untitled](/EDA/Untitled22.png)

Isı haritasına bakarak, evli ve ilkokul mezunu insanların pazarlama aramalarını en az cevaplayan grup olduğunu, üniversite mezunu bekar insanların ise en çok cevaplayan grup olduğu çıkarımında bulunabiliriz.
