# (opencv_clahe.py) OpenCV CLAHE Processor

## Proje Açıklaması

Bu proje, görüntü kontrast artırma için CLAHE (Contrast Limited Adaptive Histogram Equalization) algoritmasını kullanarak toplu görüntü işleme yapar. Sistem, her görüntü için optimal CLAHE parametrelerini otomatik olarak hesaplar ve yüksek performanslı batch processing sağlar.

Projenin temel amacı, büyük görüntü koleksiyonlarında kontrast iyileştirme işlemini hızlı ve etkili şekilde gerçekleştirmektir. Özellikle düşük kontrastlı, karanlık veya aşırı parlak görüntülerde detay kaybını önleyerek görsel kaliteyi artırır.

## Teknik Özellikler ve Algoritmalar

Sistem, öncelikle OpenCV desteğini kontrol eder ve otomatik fallback mekanizması sağlar. `analyze_image()` fonksiyonu, görüntüyü optimize edilmiş analiz yapar ve paralel işleme için hazırlar.

Görüntü analizi aşamasında, sistem LAB color space'e dönüştürme işlemi gerçekleştirir çünkü L (luminance) kanalı üzerinde CLAHE uygulanması daha iyi renk koruması sağlar. `analyze_image()` içinde brightness istatistikleri, histogram entropy, edge density ve dynamic range değerleri hesaplanır. Histogram peaks `_fast_peak_count()` ile vectorized operasyonlar kullanılarak hızlı şekilde tespit edilir.

`calculate_optimal_clip_limit()` fonksiyonu, görüntü özelliklerine dayalı adaptif parametre hesaplaması yapar. Kontrast faktörü, entropi faktörü, dinamik aralık faktörü ve parlaklık faktörü gibi dört farklı metrik ağırlıklı ortalama ile birleştirilerek optimal clip limit değeri elde edilir. Bu değer, CLAHE algoritmasının over-enhancement yapmasını önleyerek doğal görünümlü sonuçlar üretir.

Tile size hesaplaması `calculate_optimal_tile_size()` fonksiyonunda görüntü boyutu, edge density ve histogram peaks bilgilerini kullanarak yapılır. Küçük tile size daha lokalize iyileştirme sağlarken, büyük tile size daha global etki yaratır. Sistem, görüntünün detay yoğunluğuna göre bu dengeyi otomatik olarak ayarlar.

## Paralel İşleme ve Performans

`BatchProcessor` sınıfı, ThreadPoolExecutor ile multi-threading destekli toplu işleme yapar. `process_single_image()` fonksiyonu her worker thread'de bağımsız olarak çalışır ve thread-safe operasyonları gerçekleştirir. Sistem, CPU core sayısına göre optimal worker sayısını otomatik belirler ancak I/O bound operations nedeniyle genellikle 8 worker ile sınırlandırılır.

Progress tracking için tqdm kütüphanesi kullanılır ve real-time ETA hesaplaması `calculate_eta()` fonksiyonu ile sağlanır. Her işlenen görüntü için başarı durumu, ortalama CLAHE parametreleri ve işlem hızı gibi metrikler canlı olarak güncellenir. Hata durumlarında sistem ilk 5 hatayı konsola yazdırır ve işleme devam eder.

Dosya sistemi operasyonları için Path objesi kullanılır ve recursive directory traversal ile tüm alt klasörlerdeki görüntüler işlenir. Desteklenen formatlar jpg, jpeg, png, bmp, tiff, webp ve jfif'tir. Output klasör yapısı input klasörüyle aynı hiyerarşide korunur ve gerekli intermediate klasörler otomatik oluşturulur.

## Sonuç Raporu ve Kullanım

Sistem işlem tamamlandığında kapsamlı performans raporu sunar. Toplam işlenen görüntü sayısı, başarı oranı, ortalama işlem süresi ve ortalama CLAHE parametreleri raporlanır. Tipik bir işlemde 1000 görüntü için 10-15 dakika sürmektedir.

Ortalama clip limit değerleri 1.5-3.0 arasında değişir ve düşük kontrastlı görüntülerde daha yüksek değerler otomatik olarak seçilir. Tile size genellikle 8x8 ile 16x16 arasında optimize edilir ve detaylı görüntülerde daha küçük tile boyutları tercih edilir. Bu adaptif parametreler sayesinde manuel ayarlama gerektirmeden her görüntü türü için optimal sonuçlar elde edilir.

Error handling sistemi robust şekilde tasarlanmıştır ve codec sorunları veya dosya erişim hatalarında uygun fallback mekanizmaları devreye girer. Sistem, toplam hata oranının %5'in altında kalmasını hedefler ve yaygın görüntü formatları için %99+ başarı oranı sağlar.


# (marqo_multimodal.ipynb) Hierarchical Category Classification with LoRA Fine-tuning

## Projenin Amacı

Bu proje, e-ticaret ürün görsellerini hiyerarşik kategorilere sınıflandırmak için geliştirilmiş bir derin öğrenme sistemidir. Proje, ürün resimlerini L1 (ana kategori), L2 (alt kategori) ve L3 (detay kategori) olmak üzere üç seviyeli bir hiyerarşi içinde sınıflandırmayı hedefler. Sistem, Marqo e-ticaret embedding modelini temel alarak, LoRA (Low-Rank Adaptation) tekniği ile fine-tuning yaparak kategori tahminlerinde hiyerarşik tutarlılığı korur.

Temel amaç, geleneksel düz sınıflandırma yaklaşımlarından farklı olarak, kategoriler arası hiyerarşik bağımlılıkları modelleyerek daha tutarlı ve anlamlı tahminler üretmektir. Bu yaklaşım, üst seviye kategorideki bir tahminin, alt seviye kategorilerdeki tahminleri etkilemesini sağlar ve böylece mantıksal olarak tutarsız kategori kombinasyonlarının önüne geçer.

## Modifikasyonlar ve Eklenen Özellikler

Proje, önceden eğitilmiş Marqo visual embedding modeline ek olarak önemli modifikasyonlar ve yenilikler içermektedir. En kritik eklenti, `HierarchicalCascadeHead` sınıfı aracılığıyla gerçekleştirilen kademeli hiyerarşik tahmin mekanizmasıdır. Bu yaklaşımda L1 seviyesi bağımsız olarak tahmin edilir, L2 seviyesi L1 tahminlerini girdi olarak kullanır ve L3 seviyesi hem L1 hem de L2 tahminlerinden yararlanır.

LoRA adaptasyonu, `LoRALinear` sınıfı ve `inject_lora` fonksiyonu aracılığıyla vision transformerin attention ve MLP katmanlarına enjekte edilir. Bu yaklaşım, orijinal modelin ağırlıklarını dondurup, düşük-rank matrisler ile adaptasyon yaparak hesaplama maliyetini minimize eder. LoRA parametreleri (r=20, alpha=40, dropout=0.15) ile optimize edilmiş bir konfigürasyon kullanılır.

Hiyerarşik tutarlılık, `hierarchical_consistency_loss` fonksiyonu ile sağlanır. Bu fonksiyon, geleneksel cross-entropy kaybına ek olarak, hiyerarşik kısıtlamaları KL-divergence yoluyla uygular. Eğitim sırasında soft predictions kullanılırken, test aşamasında hard predictions ile tutarlılık kontrolü yapılır.

Veri önişleme aşamasında, `HierarchicalLabelEncoder` sınıfı ile seviye bazlı encoding yapılır ve `stratified_train_val_test_split` fonksiyonu ile stratified sampling uygulanarak veri dağılımının dengeli olması sağlanır. Test seti dahil üçlü veri bölünmesi yapılarak model performansının objektif değerlendirmesi mümkün kılınır.

## Hücre Hücre Çalışma Algoritmaları

İlk hücre, Google Drive bağlantısı kurar ve görsel veri setini içeren ZIP dosyasını çıkartır. `drive.mount` fonksiyonu ile Drive erişimi sağlanır, `zipfile.ZipFile` ile arşiv açılır ve dosya sayısı kontrol edilir. Bu aşama veri erişim altyapısını hazırlar.

İkinci hücre, çıkartılan klasördeki dosya sayısını kontrol eder. `os.listdir` ve `os.path.isfile` fonksiyonları ile sadece dosyalar sayılır, alt klasörler hariç tutulur. Bu kontrol, veri bütünlüğünü doğrular.

Üçüncü hücre, CSV verisini Parquet formatına dönüştüren kapsamlı veri önişleme algoritmasını içerir. `to_drive_path` fonksiyonu ile kullanıcı yolları Colab formatına çevrilir. `normalize_id` fonksiyonu ile grup ID'leri string formatına standardize edilir. Duplicate kayıtlar `CATEGORY_LEVEL` değerine göre çözümlenir, en yüksek seviyeli kayıt seçilir. Hiyerarşik etiketler L1-L4 formatında düzenlenir ve image_path eşlemesi yapılır. Son olarak pandas ile Parquet formatında kaydedilir.

Dördüncü hücre, oluşturulan Parquet dosyasının temel istatistiklerini çıkarır. `pd.read_parquet` ile veri yüklenir, `dtypes`, `isnull().sum()` ve `describe()` fonksiyonları ile veri kalitesi analiz edilir. Bu analiz, veri bütünlüğünü doğrular.

Beşinci hücre, hiyerarşik veri önişleme pipeline'ını içerir. `HierarchicalLabelEncoder` sınıfı ile seviye bazlı label encoding yapılır. Her seviye için unique etiketler çıkartılır ve numerik mapping'ler oluşturulur. `analyze_hierarchy` fonksiyonu ile veri yapısı analiz edilir, seviye kombinasyonları ve derinlik dağılımı incelenir. `stratified_train_val_test_split` fonksiyonu ile L1+L2 kombinasyonlarına göre stratified sampling uygulanır. `create_hierarchical_jsonl` fonksiyonu ile eğitim formatında JSONL dosyaları oluşturulur. Pipeline sonunda label mappings, dataset statistics ve JSONL dosyaları kaydedilir.

Altıncı hücre, Open-CLIP kütüphanesini kurarak görsel model altyapısını hazırlar.

Yedinci hücre, hiyerarşik cascade LoRA eğitim sisteminin ana implementasyonunu içerir. Colab ortam kurulumu `setup_colab_environment` fonksiyonu ile yapılır. Marqo modeli `load_marqo_model` fonksiyonu ile yüklenir ve `inject_lora` ile adaptasyon katmanları eklenir. `HierarchicalCascadeHead` sınıfı cascade mimarisini implement eder, L2 ve L3 seviyelerinin önceki seviyelerden bilgi almasını sağlar. `hierarchical_consistency_loss` fonksiyonu hiyerarşik kısıtlamaları uygular. `train_hierarchical_model` ana eğitim döngüsünü yönetir, her epoch'ta validation yapar ve en iyi modeli kaydeder. `evaluate_model` fonksiyonu detaylı metrikleri hesaplar. Eğitim AdamW optimizer, cosine scheduler ve mixed precision ile optimize edilir.

Sekizinci ve dokuzuncu hücreler, veri kontrolü ve ana eğitim fonksiyonlarını çalıştırır. `check_hierarchical_data` veri dosyalarının varlığını kontrol eder, `load_data_and_mappings` fonksiyonu ile veri yüklenir ve hierarchy mapping oluşturulur. `main` fonksiyonu eğitim konfigürasyonunu ayarlar (5 epoch, batch_size=96, lr=2e-4) ve eğitimi başlatır.

Son hücreler, kapsamlı sonuç analizi yapar. `print_detailed_console_results_with_test` fonksiyonu train/validation/test setleri için detaylı metrikleri yazdırır. `create_test_set_comparison_visualization` fonksiyonu görselleştirmeler oluşturur. `complete_analysis_with_test` bütün analiz sürecini koordine eder.

## Sonuç Çıktıları

Sistem, train/validation/test setleri üzerinde kapsamlı performans metrikleri sunar. Her hiyerarşi seviyesi (L1, L2, L3) için accuracy, precision, recall ve F1-score değerleri hesaplanır. Hiyerarşik tutarlılık oranı, kategori violasyonlarının sayısı ve oranı raporlanır. Model, training accuracy açısından yüksek performans gösterirken, validation ve test setlerinde genelleme kabiliyeti ölçülür.

Cascade mimarisi sayesinde L1 seviyesinde yüksek doğruluk elde edilir ve bu başarı alt seviyelere kademeli olarak aktarılır. LoRA adaptasyonu, orijinal modelin bilgisini korurken domain-specific özellikler öğrenir. Hiyerarşik consistency loss, mantıksal olarak tutarsız kategori kombinasyonlarını minimize eder.

Sonuç analizinde generalizasyon boşlukları (validation-test gap, training-test gap) hesaplanır ve overfitting durumu değerlendirilir. Seviye bazlı sample dağılımı, derinlik analizi ve violation oranları detaylı raporlama sağlar. Görselleştirmeler, accuracy karşılaştırmaları, consistency oranları, F1-score dağılımları ve sample istatistiklerini içerir.

Sistem, checkpoint kaydetme ile best model selection yapar ve Google Drive senkronizasyonu ile sonuçları kalıcı hale getirir. Training history, loss curves ve learning rate scheduling bilgileri kaydedilerek eğitim sürecinin takibi mümkün kılınır.