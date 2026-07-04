# GroundingDINO Image Processing Project

## Projenin Amacı

Bu proje, GroundingDINO modelini kullanarak ürün görsellerinden otomatik olarak en uygun bölgeyi tespit etmek ve kırpmak amacıyla geliştirilmiştir. Sistem, verilen kategorik bilgileri kullanarak görsellerdeki ilgili nesneleri tanımlar ve en yüksek kaliteli bölgeyi seçerek çıkarır.

## Model Modifikasyonu ve Eklenen Özellikler

Projenin en önemli yeniliği **dengeli skorlama sistemi** algoritmasıdır. Geleneksel GroundingDINO yaklaşımı sadece confidence skorunu dikkate alırken, bu sistemde hem confidence değeri hem de tespit edilen bölgenin alan büyüklüğü birleştirilerek yeni bir skorlama metriği oluşturulmuştur. `balanced_score` fonksiyonu, `W_P * confidence + W_S * normalized_area` formülü ile çalışır ve parametrik ağırlıklandırma sayesinde farklı kullanım senaryolarına göre optimize edilebilir.

Sistem ayrıca **minimum alan filtresi** uygulayarak %10'dan küçük alanları otomatik olarak elemekte ve bu sayede çok küçük veya hatalı tespitlerin sonuç üzerindeki olumsuz etkilerini önlemektedir. Koordinat sistemleri arasında dönüşüm yapan `box_cxcywh_norm_to_xyxy_pixels` algoritması, normalize edilmiş merkez-genişlik-yükseklik formatındaki kutulari piksel tabanlı x1,y1,x2,y2 koordinatlarına dönüştürürken, görsel sınırları içinde kalacak şekilde clamp işlemi uygular.

Gelişmiş kutu seçim algoritması, her tespit için hem confidence hem de alan bilgisini birleştirerek çok kriterli bir değerlendirme yapar. `compute_confidences` fonksiyonu, modelin çıktı logitlerini probability değerlerine dönüştürürken sigmoid aktivasyonu veya doğrudan clamp işlemi arasında akıllı seçim yapar. Bu sayede model çıktılarının farklı formatlarına karşı dayanıklı bir sistem elde edilir.

## Hücre Bazlı Çalışma Mantığı

İlk hücre, sistemin temel kurulumunu ve model yüklemesini gerçekleştirir. GroundingDINO modelinin konfigürasyon dosyası ve ağırlık dosyaları yüklenir, CUDA desteği kontrol edilir ve sistem kaynaklarına göre cihaz seçimi yapılır. Bu aşamada modelin hazır duruma getirilmesi ve gerekli kütüphanelerin import edilmesi sağlanır.

İkinci hücrede veri yükleme ve kontrol işlemleri yapılır. CSV dosyasından görsel yolları ve kategori bilgileri okunur, veri yapısının doğruluğu kontrol edilir ve temel istatistikler çıkarılır. `pick_prompt` fonksiyonu burada tanımlanan kategori hiyerarşisi mantığı ile çalışır ve en spesifik kategoriden en genele doğru uygun prompt seçimi yapar.

Üçüncü hücre, sistemin matematiksel altyapısını oluşturan yardımcı fonksiyonları tanımlar. `clamp01` fonksiyonu sayıları 0-1 aralığında sınırlayarak güvenli hesaplamalar sağlar. Alan hesaplamaları ve koordinat dönüşümleri bu hücredeki fonksiyonlar aracılığıyla gerçekleştirilir.

Dördüncü hücrede tanımlanan `process_one` fonksiyonu, tek bir görseli işleyen ana algoritmanın kalbidir. Bu fonksiyon, görseli yükler, GroundingDINO ile tespit yapar, her kutu için dengeli skor hesaplar, geçerli kutular arasından en yüksek skorluyu seçer ve ilgili bölgeyi kırpar. Süreç boyunca detaylı bilgi toplar ve yapılandırılmış bir rapor oluşturur.

Son hücre, toplu işleme ve loglama sistemini çalıştırır. Tüm veri seti üzerinde iterasyon yaparken gerçek zamanlı ilerleme takibi sağlar, her işlenen görsel için detaylı log kaydı tutar ve hata durumlarında sistemi ayakta tutacak exception handling mekanizmalarını devreye sokar. İşlem hızı ve kalan süre hesaplamaları ile kullanıcıya sürekli bilgi verir.

## Sistem Çıktıları

Sistem iki ana çıktı türü üretir: işlenmiş görseller ve detaylı log dosyası. İşlenmiş görseller, belirtilen output dizininde groupID_s değerini dosya adı olarak kullanarak JPG formatında saklanır. Her görsel, algoritmanın belirlediği en uygun bölgeden kırpılmış halde çıkarılır.

Log dosyası; her işlenen görsel için zaman damgası, kullanılan prompt, tespit edilen toplam kutu sayısı, geçerli kutu sayısı ve en iyi 3 tespitin detaylı skorlaması kaydedilir.

Gerçek zamanlı performans metrikleri, işleme hızı, kalan süre tahmini ve başarı oranları gibi bilgiler sürekli olarak güncellenir. Sistem, parametrik konfigürasyon ile BOX_THRESHOLD=0.1, TEXT_THRESHOLD=0.1, W_P=0.3 ve W_S=0.7 değerleri kullanır, ancak bu değerler farklı veri setleri için ayarlanabilir.

Top-3 sıralaması özelliği, her görsel için en iyi üç tespit sonucunu index, skor, confidence, alan yüzdesi ve koordinat bilgileriyle birlikte raporlar.

---

**Not**: Bu kodda en hatalı kategori olan kedi mamaları tespit edildiği için çıktılarında yalnız kedi mamaları vardır ancak 5000'e yakın görsel üzerinde test edilen kodun son hali de bunun aynısıdır, bir fark yoktur.