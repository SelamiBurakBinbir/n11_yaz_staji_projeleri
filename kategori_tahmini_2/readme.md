# Multimodal L1 Classifier

## Projenin Amacı

Bu proje, e-ticaret ürün sınıflandırması için çok modaliteli (multimodal) bir makine öğrenmesi sistemi geliştirmek amacıyla tasarlanmıştır. Sistem, önceden embedlenmiş görsel ve metin özelliklerini kullanarak ürünleri L1 seviyesinde kategorize etmeye odaklanır. Ana hedefler arasında hem görsel hem de metin bilgisini optimal şekilde birleştiren hibrit modellerin geliştirilmesi, farklı yaklaşımların performansının karşılaştırılması ve ensemble teknikleriyle doğruluğun artırılması yer almaktadır.

Projenin temel motivasyonu, geleneksel tek modaliteli yaklaşımların sınırlarını aşarak daha kapsamlı bir ürün anlayışı elde etmektir. Görsel özellikler ürünün fiziksel görünümü hakkında bilgi sağlarken, metin özellikleri ürün adı ve açıklamasından semantik bilgi çıkarır. Bu iki modaliteden gelen bilgilerin etkili kombinasyonu, daha doğru ve güvenilir sınıflandırma sonuçları hedeflenmektedir.

## Modifikasyonlar ve Eklenen Özellikler

### Gating Mekanizmalı Hibrit Füzyon

Geleneksel sabit ağırlık birleştirme yaklaşımlarından farklı olarak, sistem adaptif gating mekanizması kullanır. `MultiModalL1` sınıfı içindeki `gate` modülü, görsel ve metin özelliklerini analiz ederek her örnek için optimal karışım oranını (alpha değeri) dinamik olarak belirler. Bu mekanizma, bazı ürünler için metinsel bilgilerin daha önemli olduğu, diğerleri için görselin daha belirleyici olduğu durumları otomatik olarak yakalayabilir.

### ArcFace Tabanlı Metrik Öğrenme

Klasik softmax loss yerine ArcFace margin tabanlı loss fonksiyonu kullanılır. `ArcMarginProduct` sınıfı, eğitim sırasında sınıflar arası ayrımı artırmak için angular margin uygular ancak değerlendirme sırasında saf cosine benzerlik döndürür. Bu yaklaşım, özellik uzayında daha iyi ayrılmış sınıf temsillerini öğrenmeyi teşvik eder ve overfitting riskini azaltır.

### Class Token Attention Mekanizması

`ClassTokenAttention` modülü, her sınıf için öğrenilebilir prototype vektörler kullanarak attention tabanlı özellik iyileştirme sağlar. Bu mekanizma, giriş özellik vektörünü tüm sınıf prototipleri ile etkileşime sokar ve weighted attention çıktısı üretir. Residual bağlantılar sayesinde orijinal bilgi korunurken, sınıf-spesifik özellikler güçlendirilir.

### Kapsamlı Cache Sistemi ve Veri Yönetimi

Sistem, büyük ölçekli veri setlerinde hızlı çalışmabilmek için üç katmanlı cache sistemi uygular. Text embeddings pickle formatında, görsel özellikler numpy compressed formatta ve label matrisleri optimize edilmiş binary formatta saklanır. Bu yaklaşım, tekrarlanan çalışmalarda feature extraction sürecini tamamen atlayarak dakikalardan saniyelere düşürür.

### FAISS Tabanlı Benzerlik Arama

Geleneksel kNN yaklaşımlarını GPU hızlandırmalı FAISS kütüphanesiyle destekler. `L1FAISSSystem` sınıfı, eğitim örnekleri üzerinde IVF (Inverted File) veya HNSW (Hierarchical Navigable Small World) indeksleri oluşturarak hızlı benzerlik araması yapar. Bu sistem, model tabanlı tahminlerle FAISS tabanlı komşuluk bilgisini hibrit şekilde birleştiren ensemble yaklaşımı sunar.

### Multi-Algorithm Baseline Framework

Sistem, performans karşılaştırması için kapsamlı baseline framework içerir. Logistic Regression (text-only ve image-only), XGBoost (GPU hızlandırmalı fused features) ve weighted ensemble grid search gibi farklı yaklaşımları destekler. Bu framework, multimodal fusion yaklaşımının etkinliğini objektif şekilde değerlendirmek için referans noktaları sağlar.

## Hücre Hücre Çalışma Algoritmaları

### Veri Hazırlama ve Preprocessing Pipeline

`load_and_prepare()` fonksiyonu ile sistem, CSV dosyasını pandas kullanarak okur ve `coerce_vec()` yardımcı fonksiyonu ile her embedding'in 768 boyutlu olduğunu doğrular. `group_key_from_id()` fonksiyonu, dosya isimlerinden grup anahtarları çıkararak veri sızıntısını önlemek için kullanılır. Bu grup anahtarları, aynı ürünün farklı varyantlarının train/validation setler arasında dağılmasını engeller.

`_norm_pathlike_id()` fonksiyonu, dosya yollarını temizleyerek sadece dosya adı kökünü çıkarır. Grup anahtarı belirleme sürecinde, dosya adında underscore veya dash karakteri varsa bunlardan önceki kısmı grup anahtarı olarak kullanır. `StratifiedGroupKFold` ile veri bölümleme yapılırken, hem sınıf dengesi korunur hem de grup bütünlüğü sağlanır.

### Multimodal Fusion ve Eğitim Algoritması

`MultiModalL1` modeli içinde fusion süreci çok aşamalı olarak gerçekleşir. İlk aşamada `F.normalize()` ile görsel ve metin vektörleri L2 normalize edilir. `gate` modülü, concatenated features üzerinde sigmoid aktivasyon ile alpha katsayısını hesaplar. Bu alpha değeri, metin ve görsel özellikler arasındaki ağırlıklı kombinasyonu belirler: `z = alpha * title_features + (1 - alpha) * image_features`.

`fusion` sequential modülü, elde edilen hibrit vektörü 768 boyutundan 512 boyutuna dönüştürür. Bu süreçte LayerNorm, GELU aktivasyon ve Dropout katmanları kullanılır. `ClassTokenAttention` modülü devreye girerek, 512 boyutlu vektör ile öğrenilebilir sınıf prototipleri arasında attention hesaplar ve residual bağlantı uygular.

### ArcFace Loss ve Optimizasyon Stratejisi

`ArcMarginProduct` sınıfı, eğitim ve değerlendirme modları arasında farklı davranır. Eğitim sırasında `labels` parametresi verildiğinde, angular margin uygulayarak target sınıf için cos(θ+m) hesaplar ve diğer sınıflar için saf cosine benzerlik kullanır. `LabelSmoothingFocalLoss` ile bu logits işlenir, hem label smoothing hem de focal weight uygulanır.

`Trainer` sınıfının `train_epoch()` metodu, mixed precision training için `torch.cuda.amp.autocast()` kullanır ve gradient clipping uygular. `OneCycleLR` scheduler ile öğrenme oranı cosine annealing pattern izler. Validation aşamasında `labels=None` parametresi ile model saf cosine benzerlik döndürür ve margin uygulamaz.

### FAISS Entegrasyonu ve Hibrit Tahmin

`L1FAISSSystem` sınıfının `build_faiss()` metodu, eğitim örnekleri üzerinde fused features çıkarır ve `faiss.normalize_L2()` ile normalize eder. IVF index için `IndexIVFFlat` kullanılır ve `train()` metodu ile clustering yapılır. HNSW alternatifi için `IndexHNSWFlat` 32 bağlantılı graph oluşturur.

`predict_hybrid()` metodu, model tabanlı tahminlerle FAISS aramasını birleştirir. Model önce `_batch_logits()` ile logits üretir ve softmax uygular. Paralel olarak `_batch_fused()` ile query features çıkarır ve FAISS üzerinde `search()` yapar. En yakın k komşunun sınıf dağılımı weighted voting ile probability matrisine dönüştürülür: `hybrid_proba = alpha * model_proba + (1-alpha) * faiss_proba`.

### Ensemble ve Grid Search Optimizasyonu

`train_lr_text_only()` ve `train_lr_img_only()` fonksiyonları, tek modalite baseline'ları oluşturur. `StandardScaler` ile özellikler normalize edilir ve `LogisticRegression` multinomial solver ile eğitilir. Class imbalance için `class_weight="balanced"` parametresi kullanılır.

`train_xgb_fused()` fonksiyonu, görsel ve metin özelliklerini concatenate ederek 1536 boyutlu vektör oluşturur. XGBoost GPU acceleration için `tree_method="gpu_hist"` ve `predictor="gpu_predictor"` parametrelerini kullanır. Sample weights ile class imbalance işlenir ve early stopping ile overfitting önlenir.

`ensemble_grid_search()` fonksiyonu, farklı model kombinasyonları için grid search yapar. İki veya üç model için tüm olası ağırlık kombinasyonları (0.0 ile 1.0 arasında 0.1 step ile) test edilir. Her kombinasyon için weighted average probability hesaplanır ve validation accuracy'si karşılaştırılır. En iyi performans gösteren ağırlık kombinasyonu JSON formatında saklanır.

### Değerlendirme ve Raporlama Sistemi

`evaluate()` metodu, validation seti üzerinde model performansını değerlendirir. `_combined_scores()` ile hibrit skorlar hesaplanır ve `torch.argmax()` ile tahminler üretilir. `accuracy_score()` ve `top_k_accuracy_score()` metrikleri hesaplanır.

`_write_reports()` metodu, detaylı classification report ve confusion matrix üretir. `classification_report()` sınıf bazında precision, recall ve f1-score değerlerini hesaplar. Confusion matrix numpy formatında saklanır ve hata analizi için kullanılabilir. Alpha değerlerinin dağılımı gating mekanizmasının etkinliğini gösterir.

`test_hierarchical()` benzeri fonksiyonlar, gerçek dünya senaryolarını simüle eder ve rastgele seçilen örnekler üzerinde tahmin yapar. Sonuçlar timestamp'li CSV dosyalarında saklanır ve her tahmin için güven skorları kaydedilir.

## Sonuç Çıktıları

### Multimodal Fusion Model Performansı

ArcFace tabanlı multimodal model, validation setinde %63.48 top-1 accuracy ve %86.17 top-5 accuracy elde etmiştir. Gating mekanizması, ortalama alpha değeri 0.64 ile metin özelliklerine hafif eğilim göstermektedir. Bu değer, e-ticaret ürün sınıflandırmasında metin bilgisinin (ürün adı, açıklama) görsel bilgiden biraz daha discriminative olduğunu gösterir. Training süreci boyunca alpha değerinin 0.27'den 0.65'e evrilmesi, modelin optimal modalite dengesini öğrendiğini işaret eder.

Label smoothing (0.05) ve focal loss (gamma=1.5) kombinasyonu, class imbalance problemini etkili şekilde ele almıştır. En büyük sınıf olan "Yapı Market & BahÃ§e" (7946 örnek) ile en küçük sınıflar arasındaki performance gap minimize edilmiştir. Classification report'ta görülen sınıf bazında precision/recall değerleri, sistem dengesinin başarıyla sağlandığını göstermektedir.

### FAISS Hibrit Sistem Sonuçları

FAISS entegrasyonu, model-only yaklaşımına göre %3.00 puanlık iyileştirme sağlamıştır (%72.42'den %75.42'ye). IVF index ile 57,743 training örneği üzerinde 512 boyutlu normalized features kullanılmıştır. k=10 komşuluk araması ve alpha=0.6 model ağırlığı optimal kombinasyon olarak bulunmuştur. Bu sonuç, local neighborhood bilgisinin global model predictions'ı tamamlayıcı etkisini doğrulamaktadır.

FAISS system memory footprint'i yaklaşık 120MB (57K × 512 × 4 byte) olarak hesaplanmıştır. Query time performance'ı batch size 2048 için yaklaşık 50ms/batch seviyesindedir. Index building süresi A100 GPU'da 15 saniye civarında gerçekleşmektedir. Bu metrikler, production deployment için acceptable seviyededir.

### Baseline Model Karşılaştırmaları

XGBoost fused model, %80.68 top-1 accuracy ile en yüksek single-model performansını sergilemiştir. GPU acceleration (tree_method="gpu_hist") sayesinde 1536 boyutlu features üzerinde 2000 estimator ile training yaklaşık 3 dakika sürmüştür. Early stopping 100 round sonunda devreye girmiş ve overfitting önlenmiştir.

Text-only Logistic Regression %68.00, image-only Logistic Regression daha düşük performance göstermiştir. Bu fark, metin modalitesinin e-ticaret sınıflandırmasında görsel modaliteden daha bilgi taşıdığını desteklemektedir. Ensemble grid search sonucunda, optimal ağırlık kombinasyonu %70.76 accuracy ile %60 text + %40 fusion olarak bulunmuştur.

### Training Infrastructure ve Scalability

A100 GPU üzerinde batch size 256 ile training, epoch başına yaklaşık 2.5 dakika sürmektedir. Mixed precision training ile memory usage 18GB seviyesinde kalmakta ve 40 GB VRAM'in yaklaşık %45'ini kullanmaktadır. OneCycleLR scheduler ile 40 epoch'ta convergence sağlanmış, early stopping patience 10 ile overfitting önlenmiştir.

Cache system efficiency oldukça yüksek seviyededir. İlk çalışmada feature extraction 45 dakika sürerken, sonraki çalışmalarda cache loading 2 dakikada tamamlanmaktadır. Bu %95'lik süre tasarrufu, iterative model development için kritik öneme sahiptir. NPZ compressed format ile disk storage %40 azaltılmıştır.

### Test Sonuçlarının Güvenilirlik Analizi
Bu sistemde de kritik bir veri sızıntısı problemi bulunmaktadır. test_hierarchical() fonksiyonunda test örnekleri np.random.choice() ile tüm veri setinden rastgele seçilmekte, bu da train setinde bulunan örneklerin test setinde de yer almasına neden olabilmektedir. Benzer şekilde, compare_on_val() fonksiyonunda FAISS hibrit sistem aynı validation seti üzerinde değerlendirilmektedir.

Bu durum, raporlanan performans değerlerinin gerçek genelleme kapasitesini yansıtmamasına yol açmaktadır. Validation accuracy %63.48, FAISS hibrit accuracy %75.42 ve XGBoost baseline %80.68 gibi değerler, model güvenilirlik açısından şüpheli görülmelidir çünkü sistem bu görüntüleri eğitim sürecinde "ezberlemek" için fırsata sahip olmuştur.

Gerçek, tamamen unseen test setinde yapılacak değerlendirmelerde bu accuracy değerlerinin muhtemelen %15-25 daha düşük çıkacağı öngörülmelidir. Bu durum, makine öğrenmesinde "data leakage" problemi olarak bilinir ve production ortamında ciddi performans düşüşlerine neden olabilir. Dolayısıyla, raporlanan tüm metrikler sistemin üst sınır performansını göstermekte olup, gerçek dünya uygulamalarında çok daha konservatif sonuçlar beklenmektedir.