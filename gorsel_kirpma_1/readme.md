# Florence-2 Toplu Nesne Algılama Sistemi

## Proje Amacı

Bu proje, Microsoft'un Florence-2 vision-language modelini kullanarak çok sayıda görselde belirli nesneleri otomatik olarak tespit eden, bu nesnelerin konumlarını sınırlayıcı kutularla işaretleyen ve sonuçları görsel ve metin formatında raporlayan bir toplu işlem sistemidir.

## Model Modifikasyonu ve Eklenen Özellikler

Bu kodda herhangi bir fine-tune, model eğitimi veya özel model modifikasyonu bulunmamaktadır. Kod, Microsoft'un orijinal Florence-2-base modelini olduğu gibi kullanmaktadır - sadece Hugging Face transformers kütüphanesi üzerinden pre-trained modeli yükleyip inference yapmaktadır. Kodun eklediği özellikler tamamen model dışında, uygulama katmanındadır: toplu dosya işleme sistemi, görselleştirme (matplotlib ile sınırlayıcı kutu çizimi) ve raporlama sistemi (analiz çıktıları). Model seviyesinde hiçbir değişiklik yoktur; Florence-2'nin mevcut "caption-to-phrase grounding" task'ını kullanarak phrase-based nesne lokalizasyonu gerçekleştirmektedir.

## Hücre Bazlı Çalışma Mantığı

### Hücre 0: Yollar & Klasör Hazırlığı

Bu hücre sistemin temel konfigürasyonunu gerçekleştirir. Pathlib kullanarak platform bağımsız dosya yolu yönetimi sağlar ve tüm proje dizinlerini tanımlar. BASE_DIR ana çalışma dizinini, LABELS_FILE giriş verilerinin bulunduğu dosyayı, OUT_DIR çıktı görsellerinin kaydedileceği klasörü ve ANALYSIS_FILE analiz raporunun yazılacağı dosyayı belirler. Klasör yapısını oluşturur ve eksik dizinleri otomatik olarak yaratır. Bu aşama, tüm sonraki işlemlerin düzgün çalışması için gerekli altyapıyı hazırlar.

### Hücre 1: Model Kurulumu & Yardımcı Fonksiyonlar

Bu hücre Florence-2 modelinin yüklenmesi ve inference için gerekli yardımcı fonksiyonların tanımlanması işlemlerini gerçekleştirir. İlk olarak sistem kaynaklarını analiz ederek CUDA varlığını kontrol eder ve uygun device seçimini yapar. Microsoft Florence-2-base modelini ve processor'ını trust_remote_code=True parametresiyle güvenli şekilde yükler. _safe_name() fonksiyonu Türkçe karakterleri koruyarak dosya isimlerini zararsız hale getirir. florence_run() fonksiyonu phrase-grounding modunda inference gerçekleştirirken sadece model çalıştırma süresini hassas bir şekilde ölçer. save_result() fonksiyonu matplotlib kullanarak algılanan nesneleri kırmızı sınırlayıcı kutularla işaretleyerek görselleştirir ve yüksek çözünürlükte kaydeder.

### Hücre 2: Toplu İnference Döngüsü

Bu hücre asıl toplu işlem mantığını uygular. Labels dosyasını satır satır okuyarak her entry için otomatik processing gerçekleştirir. Her satırda ürün ID, aranacak phrase ve görsel yolu bilgilerini pipe (|) karakteriyle ayrıştırır. Görseli PIL ile açıp RGB formatına dönüştürdükten sonra florence_run() fonksiyonu ile inference yapar. Başarılı algılama durumunda bulunan nesneleri sınırlayıcı kutularla işaretleyerek outputs klasörüne kaydeder, eğer nesne bulunamazsa "WARNING_EMPTY" etiketiyle işaretler. Her işlem için ürün ID, bulunan nesneler, işlem süresi ve çıktı dosya yolu bilgilerini analysis dosyasına yazarak detaylı bir log tutar. Hata durumlarında (görsel açılamama, format hataları) işlemi atlar ve konsola hata mesajı yazdırır.

## Sistem Çıktıları

### Görsel Çıktılar (outputs_2/ klasörü)

Sistem her başarılı analiz için işaretlenmiş bir görsel üretir. Bu görseller JPG formatında 150 DPI yüksek çözünürlükle kaydedilir ve `{ÜRÜN_ID}__{TEMİZLENMİŞ_PHRASE}.jpg` şablonuyla isimlendirilir. Görsel içerik olarak orijinal görsel üzerinde algılanan nesnelerin etrafına kırmızı sınırlayıcı kutular çizilir ve her kutunun üstünde beyaz arka planla algılanan nesne ismi gösterilir. Örneğin `PROD001__kahve_fincani.jpg` gibi dosyalar oluşturulur.

### Analiz Raporu (output_analysis_2.txt)

Her işlem için bir satır analiz verisi içeren metin dosyası oluşturulur. Format `ÜRÜN_ID | BULUNAN_NESNELER | İŞLEM_SÜRESİ | ÇIKTI_DOSYASI` şeklindedir. 

- Başarılı algılama durumunda: `PROD001 | kahve fincanı, tabak | 245.67ms | outputs_2\PROD001__kahve_fincani.jpg`
- Boş sonuç durumunda: `PROD002 | WARNING_EMPTY | 198.45ms | outputs_2\PROD002__kedi.jpg`
- Çoklu nesne algılandığında: `PROD003 | araba, trafik lambası, yol | 312.89ms | outputs_2\PROD003__araba.jpg`

### Konsol Çıktısı ve Monitoring

İşlem sırasında anlık durum bilgileri konsola yazdırılır ve real-time monitoring sağlanır. Her işlem için:

- Başarılı sonuçlar: `[PROD001] tamam → 2 kutu, 245.67 ms, PROD001__kahve_fincani.jpg`
- Boş algılama durumları: `[PROD002] tamam → 0 kutu, 198.45 ms, PROD002__kedi.jpg`
- Hata mesajları: `[PROD003] Görsel açılamadı → [Errno 2] No such file or directory`

Bu konsol çıktısı sayesinde işlemin ilerleyişi ve her adımdaki sonuçlar anında takip edilebilir.