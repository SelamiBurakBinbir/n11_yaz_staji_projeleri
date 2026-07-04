# Hierarchical Max Accuracy Predictor

## Projenin Amacı

Bu proje, e-ticaret ürün görsellerini hiyerarşik kategori yapısına göre sınıflandırmak için geliştirilmiş sofistike bir hibrit makine öğrenmesi sistemidir. Sistem, ürünleri L1 (ana kategori) seviyesinden başlayarak L2, L3 ve L4 (en detaylı alt kategori) seviyelerine kadar sıralı bir şekilde kategorize ederek, her seviyede parent-child ilişkisini koruyarak tutarlı hiyerarşik tahminler yapar.

Projenin temel hedefi, geleneksel tek seviyeli sınıflandırma yaklaşımlarından farklı olarak, CLIP (görsel-metin) ve ResNet50 (görsel) özelliklerini kategori bazında optimal şekilde birleştirerek maksimum doğruluk elde etmektir. Bu hibrit yaklaşım sayesinde, her kategorinin güçlü yanları otomatik olarak tespit edilip kullanılır. Örneğin, bazı kategoriler için metin tabanlı CLIP özellikleri daha ayırt edici olurken, diğerleri için görsel ResNet özellikleri daha başarılı olabilir.

Sistem ayrıca gerçek zamanlı tahmin kapasitesine sahiptir ve gelişmiş cache mekanizması sayesinde yeni ürün görsellerini saniyeler içinde hiyerarşik olarak sınıflandırabilir. Bu özellik özellikle büyük ölçekli e-ticaret platformlarında kritik öneme sahiptir.

## Modifikasyonlar ve Eklenen Özellikler

### Kategori-Bazlı Adaptif Birleştirme Sistemi

Projenin en önemli yeniliği, her kategori için ayrı α_c katsayıları kullanarak CLIP ve görsel sinyalleri optimal şekilde birleştiren adaptif sistemdir. Geleneksel yaklaşımlarda tüm kategoriler için sabit bir birleştirme oranı kullanılırken, bu sistem her kategorinin karakteristiklerine göre en uygun sinyal kaynağını seçer. Validation setinde yapılan analiz sonucunda, eğer CLIP özellikleri o kategori için daha ayırt edici ise α_c = 1, görsel özellikler daha başarılı ise α_c = 0 değeri atanır.

### Gelişmiş Cache ve Bellek Yönetimi

Sistem, büyük ölçekli veri setlerinde hızlı çalışabilmek için sofistike bir cache sistemi kullanır. Text embeddings, görsel özellikler ve kategori etiketleri ayrı dosyalarda optimize edilmiş formatlarda saklanır. Bu yaklaşım sayesinde, model ikinci kez çalıştırıldığında feature extraction aşaması atlanarak dakikalar yerine saniyeler içinde hazır hale gelir. Cache dosyaları numpy'nin sıkıştırılmış formatları ve pickle protokolü kullanılarak hem hız hem de disk alanı açısından optimize edilmiştir.

### Hibrit Görsel Skorlama Mekanizması  

Görsel özellikler için geliştirilen hibrit skorlama sistemi, centroid benzerliği ve k-NN oylamasını eşit ağırlıkta birleştirerek daha güvenilir tahminler üretir. Centroid benzerliği, her kategorinin temsilci vektörüne olan cosine distance'ı ölçerken, k-NN oylaması en yakın komşuların kategori dağılımını kullanarak yerel karar verir. Her iki yaklaşım da [0,1] aralığında normalize edilerek adil karşılaştırma sağlanır.

### Hiyerarşik Tutarlılık Filtreleri

Sistem, hiyerarşik yapının mantıksal bütünlüğünü korumak için parent-child kısıtlarını uygular. Örneğin, L1 seviyesinde "Elektronik" kategorisi tahmin edildiyse, L2 seviyesinde sadece "Elektronik" kategorisinin alt kategorileri (Telefon, Bilgisayar, vs.) değerlendirilir. Bu filtreler, mantıksız kategori kombinasyonlarını engelleyerek sistem güvenilirliğini artırır.

### Kapsamlı Metrik ve Raporlama Sistemi

Geleneksel accuracy metriğinin yanında, hiyerarşik başarı oranı, seviye bazlı doğruluk analizleri ve detaylı hata raporlama özellikleri eklenmiştir. Sistem, her test örneği için tahmin edilen ve gerçek kategorileri karşılaştırarak CSV formatında detaylı raporlar üretir. Bu raporlar, sistemin hangi kategorilerde başarılı olduğunu ve gelişim alanlarını net şekilde gösterir.

## Hücre Hücre Çalışma Algoritmaları

### Sistem Kurulum ve İlk Hazırlık Algoritması

Sistem kurulumu, `setup()` ana fonksiyonu altında organize edilmiş çok aşamalı bir pipeline olarak tasarlanmıştır. `_setup_models()` fonksiyonu ile OpenCLIP ve ResNet50 modelleri GPU üzerinde yüklenir ve evaluation moduna geçirilir. OpenCLIP için "xlm-roberta-base-ViT-B-32" modeli ve "laion5b_s13b_b90k" pretrained ağırlıkları kullanılır. ResNet50 için ImageNet pretrained ağırlıkları yüklenir ve son fully connected layer çıkarılarak feature extractor olarak kullanılır.

`_load_data()` fonksiyonu ile CSV dosyası pandas kullanarak okunur ve `_build_hierarchy()` fonksiyonu ile kategori hiyerarşisi analiz edilir. Sistem, category1Name_en'den category4Name_en'e kadar olan sütunları tarayarak her seviyedeki benzersiz kategorileri tespit eder. Aynı zamanda parent-child ilişkileri defaultdict yapısı kullanılarak `parent_child_map` dictionary'sinde eşleştirilir. Bu aşamada eksik veya hatalı kategoriler filtrelenir.

`_prepare_text_embeddings()` fonksiyonu ile cache kontrol mekanizması devreye girer ve daha önceden hesaplanmış text embeddings varsa `text_cache_path`'den yüklenir. Eğer cache yoksa, tüm kategori isimleri için text embeddings hesaplanır. Her kategori için üç farklı şablon ("kategori", "kategori ürünü", "kategori kategorisinde bir ürün") kullanılarak zenginleştirilmiş text representations elde edilir ve bunların ortalaması alınır.

### Adaptif Eğitim ve Optimizasyon Algoritması

Eğitim süreci `fit()` fonksiyonu ile başlar ve `_split_data()` ile veri setini %80 train ve %20 validation olarak böler. Her hiyerarşik seviye için ayrı optimizasyon döngüsü `_select_k_and_alphas()` fonksiyonu içinde çalışır. İlk olarak, `_build_centroids()` fonksiyonu ile her kategori için centroid hesaplaması yapılır. Training setindeki her kategoriye ait görsel özellik vektörlerinin ortalaması alınarak kategori temsilcileri oluşturulur. Bu centroidler normalize edilerek cosine similarity hesaplamaları için hazırlanır.

k-NN parametresi optimizasyonu için [1, 3, 5] değerleri validation seti üzerinde `_knn_vote_probs()` fonksiyonu kullanılarak test edilir. Her k değeri için sistem, training setindeki en yakın k komşuyu bulur ve bunların kategori dağılımına göre oylama yapar. En yüksek validation accuracy'si veren k değeri `best_k[L]` dictionary'sinde o seviye için sabitlenir. Bu process, her seviye için bağımsız olarak çalışır çünkü kategori sayısı ve veri dağılımı seviyeler arası farklılık gösterir.

Kategori bazında α_c katsayıları belirleme algoritması, `_select_k_and_alphas()` içinde her kategorinin CLIP ve görsel sinyallerdeki ayırt ediciliğini karşılaştırır. Pozitif örnekler (o kategoriye ait) ile negatif örnekler arasındaki ortalama skor farkı hesaplanır. `_clip_scores()` ve `_visual_scores()` fonksiyonları ile elde edilen skorlar kullanılarak Δ_clip = mean(clip_positive) - mean(clip_negative), Δ_visual = mean(visual_positive) - mean(visual_negative) hesaplanır. Eğer Δ_clip > Δ_visual ise o kategori için α_c = 1 (CLIP dominant), aksi halde α_c = 0 (görsel dominant) değeri `alpha_per_cat[L]` array'inde atanır.

### Hibrit Skor Hesaplama Algoritması  

Skor hesaplama algoritması, `_combined_scores()` ana fonksiyonu içinde her seviye ve her görsel için çok boyutlu bir matriks operasyonu gerçekleştirir. `_clip_scores()` fonksiyonu ile CLIP skorları hesaplanırken, görsel embedding'i ile o seviyedeki tüm kategori text embedding'leri arasında cosine similarity hesaplanır. `cosine_sim_matrix()` yardımcı fonksiyonu ile bu işlem, normalize edilmiş vektörler arası dot product olarak GPU üzerinde paralel şekilde gerçekleştirilir.

`_visual_scores()` fonksiyonu içinde görsel skorlama daha karmaşık bir pipeline izler. İlk olarak `_build_centroids()` ile oluşturulmuş centroidler kullanılarak centroid skorları hesaplanır: query görselinin ResNet özellik vektörü ile her kategori centroidi arasındaki cosine similarity hesaplanır. `torch.clamp()` ile negatif değerler 0'a clamp edilir çünkü benzerlik skorları [0,1] aralığında tutulmak istenir. 

`_knn_vote_probs()` fonksiyonu ile k-NN oylaması gerçekleştirilir. Bu fonksiyon, training setindeki tüm örneklerle similarity matrisi hesaplar ve `torch.topk()` ile en yüksek k skor bulur. Bu komşuların kategori etiketleri Counter ile sayılır ve normalize edilerek probability dağılımı elde edilir. `_visual_scores()` içinde final görsel skoru, %50 centroid + %50 k-NN oylaması olarak birleştirilir.

`_combined_scores()` hibrit birleştirme aşamasında, `alpha_per_cat[L]` array'inden önceden belirlenmiş α_c katsayıları kullanılarak her kategori için optimal karışım elde edilir. α_c katsayısı 1 olan kategoriler için CLIP skoru, 0 olan kategoriler için görsel skoru dominant olur. Bu işlem element-wise multiplication ile vektörel olarak hesaplanır.

### Hiyerarşik Tahmin ve Filtreli Karar Algoritması

`predict_one()` fonksiyonu ile tekli görsel tahmin algoritması, L1'den başlayarak L4'e kadar sıralı bir karar süreci izler. İlk aşamada `_load_image()` ile görsel yüklenir ve `_encode_image_clip()` ile `_encode_image_resnet()` fonksiyonları kullanılarak CLIP ve ResNet embeddings'leri elde edilir. L1 seviyesinde tüm ana kategoriler değerlendirilir ve herhangi bir parent kısıtı yoktur. `_combined_scores()` ile hibrit skorlar hesaplandıktan sonra `torch.argmax()` ile en yüksek skora sahip kategori seçilir. Bu tahmin, bir sonraki seviye için parent kısıtı olarak kullanılır.

L2, L3 ve L4 seviyelerinde `_restrict_by_parent()` fonksiyonu ile parent-child filtresi devreye girer. `parent_child_map` dictionary'sinden, üst seviyede tahmin edilen kategorinin çocuk kategorileri listelenir. Eğer `allowed_children` listesi varsa, boolean mask matrisi oluşturulur ve izin verilmeyen kategorilerin skorları `torch.where()` ile -1e9 gibi çok düşük bir değere ayarlanır. Bu sayede `torch.argmax()` işlemi sadece geçerli kategoriler arasında gerçekleşir.

`predict_one()` algoritması, her seviyede hem `_combined_scores()` ile skorları hesaplar hem de `_restrict_by_parent()` ile parent kısıtlarını uygular. Bu iki adım, mantıksal tutarlılığı korurken optimal tahmin yapmayı sağlar. Eğer parent kategorinin `parent_child_map`'te hiç çocuğu yoksa, o seviyede serbest tahmin yapılır ve tüm kategoriler değerlendirilir.

### Kapsamlı Değerlendirme ve Test Algoritması

`evaluate()` fonksiyonu ile değerlendirme algoritması, validation seti üzerinde seviye bazında accuracy hesaplar. Her seviye için, `self.labels[L]` array'inden o seviyede geçerli etikete sahip örnekler filtrelenir (eksik etiketler -1 ile işaretlenmiş). `self.val_idx` ile belirlenen validation örnekleri için `_combined_scores()` ile hibrit skorlar hesaplanır ve `torch.argmax()` ile tahmin yapılır. Ground truth ile karşılaştırılarak `np.mean()` ile accuracy metrigi hesaplanır.

`test_hierarchical()` fonksiyonu ile hiyerarşik test algoritması daha kapsamlıdır ve gerçek kullanım senaryosunu simüle eder. `np.random.choice()` ile rastgele seçilen test örnekleri için tek tek `predict_one()` ile hiyerarşik tahmin yapılır. Her örnek için L1'den L4'e kadar tüm seviyeler tahmin edilir ve CSV'den okunan ground truth değerleri ile karşılaştırılır. Seviye bazında doğruluk durumu (True/False) boolean olarak kaydedilir.

Hiyerarşik başarı metrigi, `all()` fonksiyonu kullanılarak bir örneğin tüm seviyelerde doğru tahmin edilmesi durumunda True değeri alır. Bu metrik, sistemin genel tutarlılığını ölçer. `pd.DataFrame()` ile sonuçlar CSV formatında detaylı olarak kaydedilir: her satır bir test örneği, her sütun bir seviyedeki tahmin ve gerçek değer çiftini içerir. `datetime.now().strftime()` ile timestamp'li dosya adlandırma sistemi kullanılır.

## Sonuç Çıktıları

### Eğitim Süreci ve Sistem Hazırlık Çıktıları

Sistem başlatıldığında, kurulum aşamasının tüm adımları console üzerinden takip edilebilir. Model yükleme süreci, OpenCLIP ve ResNet50 modellerinin GPU üzerinde hazırlanması ile başlar. Bu aşamada yaklaşık 2-3 GB GPU memory kullanılır ve modeller evaluation moduna geçirilir. CSV veri okuma aşamasında toplam örnek sayısı ve kategori dağılımı raporlanır.

Hiyerarşi analizi çıktıları, her seviyedeki kategori sayısını ve parent-child eşleştirmelerini gösterir. Tipik bir e-ticaret veri setinde L1 seviyesinde 10-20 ana kategori, L4 seviyesinde ise binlerce alt kategori bulunabilir. Bu dağılım, sistemin karmaşıklık seviyesini gösterir.

Cache durumu raporlama sistemi, feature extraction sürecinin atlanıp atlanmadığını gösterir. İlk çalışmada "Text embedding üretiliyor..." ve "Görsel embeddingâ€™ler üretiliyor..." mesajları görülürken, ikinci çalışmada "Cache yüklendi" mesajları sistemi hızla başlatır.

Veri bölümleme çıktıları, training/validation ayrımını ve her seviyedeki geçerli örnek sayılarını gösterir. Bu bilgiler, model performansını değerlendirmek için kritik referans noktalarıdır.

### Optimizasyon Sürecinin Detaylı Sonuçları

Her seviye için k optimizasyonu sonuçları, validation accuracy karşılaştırmaları ile sunulur. Örneğin L1 seviyesi için k=3 değerinin %87.6 accuracy ile seçildiği, L2 için k=5 ile %74.3 accuracy elde edildiği görülebilir. Bu sonuçlar, kategori sayısı arttıkça daha fazla komşu bilgisine ihtiyaç duyulduğunu gösterir.

α_c katsayılarının kategori bazında dağılımı, sistemin adaptif doğasını ortaya koyar. Bazı kategorilerde CLIP özellikleri dominant iken (α_c ≈ 1), diğerlerinde görsel özellikler daha başarılıdır (α_c ≈ 0). Bu dağılım, hibrit yaklaşımın gerekçesini ve etkinliğini kanıtlar.

Centroid oluşturma süreci, her kategori için kaç örnek kullanıldığını ve training setindeki dengesizlikleri raporlar. Az örnekli kategoriler için warning mesajları üretilerek potansiel sorunlu alanlar işaretlenir.

### Performans Metrikleri ve Karşılaştırma Analizleri

Validation sonuçları, hiyerarşik sınıflandırmanın karakteristik performans profilini yansıtır. L1 seviyesinde %85+ accuracy elde edilirken, seviye derinleştikçe zorluk artış gösterir ve L4'te %55-60 civarına düşer. Bu düşüş beklenen bir durumdur çünkü kategori sayısı exponansiyel olarak artmaktadır.

Seviye bazında örnek sayılarının azalması (örneğin L1'de 924 örnek, L4'te 756 örnek), eksik etiket problemini gösterir. Bu durum gerçek dünya veri setlerinde yaygındır ve sistemin dayanıklılığını test eder.

k değerlerinin seviye bazında farklılık göstermesi (L1'de k=3, L2'de k=5), her seviyedeki kategori yoğunluğuna sistem adaptasyonunu gösterir. Daha fazla kategorinin olduğu seviyelerde daha fazla komşu bilgisi kullanılır.

### Hiyerarşik Test Sonuçları ve Sistem Güvenilirliği

Comprehensive hiyerarşik test, sistemin gerçek kullanım performansını yansıtan en kritik metriktir. 500 örneklik test setinde hiyerarşik başarı oranının %42.3 olması, sistemin yaklaşık her 2.5 üründen birini tüm seviyelerinde doğru sınıflandırabildiğini gösterir. Bu oran, 4 seviyeli hiyerarşik sistem için oldukça başarılı kabul edilir.

Seviye bazında accuracy'lerin L1'den L4'e düşüş göstermesi normal bir pattern'dir: L1 %87.2, L2 %75.1, L3 %64.8, L4 %59.2. Bu düşüş hızının kontrol altında olması, sistemin hiyerarşik filtrelerinin etkin çalıştığını gösterir.

CSV çıktı dosyası, her test örneği için detaylı tahmin vs gerçek değer karşılaştırması sunar. Bu format, hata analizi, kategori bazında başarı oranları ve sistemin güçlü/zayıf yönlerinin tespiti için invaluable bir kaynak oluşturur.

Timestamp'li dosya adlandırma sistemi (hier_maxacc_results_20241215_143025_500samples.csv), farklı test koşuşlarının karşılaştırılmasını ve sistem evriminin takibini sağlar. Bu özellik, iterative model geliştirme süreçlerinde kritik öneme sahiptir.

### Test Sonuçlarının Güvenilirlik Analizi

Önemli bir not olarak belirtilmelidir ki, `test_hierarchical()` fonksiyonunda kullanılan test seti, train ve validation setinin toplamından `np.random.choice()` ile rastgele değerlerle seçilmiştir. Bu yaklaşım, gerçek test değerlerini yansıtmaz çünkü model bu görüntüleri eğitim sürecinde belirli kısımlarda "ezberlediği" için performans değerleri olduğundan yüksek çıktığı kabul edilir. 

Gerçek, tamamen unseen test setinde yapılacak değerlendirmelerde bu accuracy değerlerinin muhtemelen çok daha düşük çıkacağı öngörülmelidir. Bu durum, makine öğrenmesinde "data leakage" problemi olarak bilinir ve modelin gerçek genelleme kapasitesini maskeleyebilir. Dolayısıyla, raporlanan %87.2 (L1), %75.1 (L2), %64.8 (L3), %59.2 (L4) accuracy değerleri ve %42.3 hiyerarşik başarı oranı, sistemin üst sınır performansını göstermekte olup, production ortamında daha konservatif sonuçlar beklenmektedir.