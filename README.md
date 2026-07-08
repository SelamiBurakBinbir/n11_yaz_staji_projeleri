# E-Ticaret Ürün Görselleri İçin Yapay Zekâ Projeleri

Bu repo, e-ticaret ürün görselleri üzerinde geliştirilen nesne tespiti, ürün kırpma ve kategori tahmini çalışmalarını içerir. Çalışmalar genel olarak iki ana probleme odaklanır:

1. Ürün görsellerinde nesne tespiti ve otomatik kırpma
2. Ürün görsellerinden hiyerarşik kategori tahmini

Repo içerisinde 5 ana proje/modül bulunmaktadır. Her klasörde ilgili kod dosyaları, kısa proje açıklaması ve örnek çıktı dosyaları yer alır.

## Proje Yapısı

```text
.
├── gorsel_kirpma_1
│   ├── florence2.py
│   ├── ornek_output.png
│   └── readme.md
├── gorsel_kirpma_2
│   ├── grounding_dino.ipynb
│   ├── ornek_output.png
│   └── readme.md
├── kategori_tahmini_1
│   ├── open-clip_resnet_hibrit.py
│   ├── ornek_output.txt
│   └── readme.md
├── kategori_tahmini_2
│   ├── mclip_multimodal.ipynb
│   ├── ornek_iyilestirme_ozeti.png
│   ├── ornek_output.png
│   └── readme.md
└── kategori_tahmini_3
    ├── marqo_multimodal.ipynb
    ├── opencv_clahe.py
    ├── ornek_test_sonucu_1.png
    ├── ornek_test_sonucu_2.png
    └── readme.md
```

## Projeler

### 1. Florence-2 Toplu Nesne Algılama

Klasör: `gorsel_kirpma_1`

Bu proje, Microsoft Florence-2 modelini kullanarak görsellerde verilen metinsel ifadeye karşılık gelen nesneleri tespit eder. Model, `caption-to-phrase grounding` görevi ile çalışır. Her görsel için sınırlayıcı kutular üretilir, sonuçlar görsel olarak kaydedilir ve işlem bilgileri metin çıktısı olarak raporlanır.

Kullanılan başlıca teknolojiler:

- Florence-2
- Hugging Face Transformers
- PyTorch
- PIL
- Matplotlib

Örnek çıktı:

<img width="1700" height="883" alt="ornek_output" src="https://github.com/user-attachments/assets/f44ac633-5d39-47bc-9a8f-92d444a05176" />

---

### 2. GroundingDINO ile Ürün Görseli Kırpma

Klasör: `gorsel_kirpma_2`

Bu proje, GroundingDINO modelini kullanarak ürün görsellerinde en uygun ürün bölgesini tespit eder ve görseli otomatik olarak kırpar. Sistem, yalnızca model güven skorunu değil, tespit edilen alanın büyüklüğünü de dikkate alan özel bir dengeli skor mekanizması kullanır.

Temel skor mantığı:

```text
score = W_P * confidence + W_S * normalized_area
```

Bu sayede çok küçük veya ürünün yalnızca bir parçasını içeren bölgeler yerine, ürünün tamamını daha iyi temsil eden alanların seçilmesi hedeflenir.

Kullanılan başlıca teknolojiler:

- GroundingDINO
- PyTorch
- OpenCV
- Pandas
- NumPy

Örnek çıktı:

<img width="830" height="890" alt="ornek_output" src="https://github.com/user-attachments/assets/efa107d6-69c0-4af1-8cc0-41595228fa9a" />

---

### 3. Open-CLIP + ResNet Hibrit Hiyerarşik Kategori Tahmini

Klasör: `kategori_tahmini_1`

Bu proje, ürün görsellerinden hiyerarşik kategori tahmini yapmak için Open-CLIP ve ResNet50 modellerini birlikte kullanır. Open-CLIP ile görsel-metin benzerliği hesaplanırken, ResNet50 ile görsel özellikler çıkarılır. Daha sonra centroid, k-NN ve kategori bazlı ağırlıklandırma yöntemleriyle tahminler birleştirilir.

Kategori tahmini L1, L2, L3 ve L4 seviyelerinde yapılır. Sistem, üst kategori tahminlerini alt kategori seçimlerinde kısıtlayıcı olarak kullanarak hiyerarşik tutarlılığı artırmayı hedefler.

Kullanılan başlıca teknolojiler:

- Open-CLIP
- ResNet50
- PyTorch
- Torchvision
- k-NN
- Cosine similarity
- Pandas / NumPy

Örnek çıktı:

```text
Kurulum başlıyor...
OpenCLIP yükleniyor...
ResNet50 yükleniyor...
Modeller hazır.
CSV yüklendi: 4920 satır
Hiyerarşi oluşturuluyor...
  L1: 64 kategori
  L2: 282 kategori
  L3: 753 kategori
  L4: 292 kategori
  Parent-child L1→L2: 64 parent
  Parent-child L2→L3: 282 parent
  Parent-child L3→L4: 149 parent
Text embedding üretiliyor...
Text embed cache kaydedildi.
Görsel embedding’ler üretiliyor (tüm dataset için)...
Encode images: 100%|███████████████████████████████████████████████████████████████| 4920/4920 [05:03<00:00, 16.20it/s]
Görsel embed cache kaydedildi: 4920 örnek
Label matrisleri hazırlanıyor...
Label cache kaydedildi.
Kurulum tamam.
Split -> train:3936 val:984 test:0
L1 ayarlanıyor (k ve α_c)...
L2 ayarlanıyor (k ve α_c)...
L3 ayarlanıyor (k ve α_c)...
L4 ayarlanıyor (k ve α_c)...

Değerlendirme (validation üzerinde)...
  L1: acc=0.761 (n=984)  | k=1
  L2: acc=0.696 (n=984)  | k=1
  L3: acc=0.640 (n=984)  | k=1
  L4: acc=0.664 (n=259)  | k=1

Hiyerarşik test başlıyor...
Testing: 100%|█████████████████████████████████████████████████████████████████████| 4920/4920 [05:52<00:00, 13.95it/s]

====================================================================================================
HİYERARŞİK TEST RAPORU - 4920 ÖRNEK
  L1: 0.943  (n=4920)
  L2: 0.889  (n=4920)
  L3: 0.834  (n=4920)
  L4: 0.866  (n=1309)
Hiyerarşik Başarı: 0.826  (4062/4920)
====================================================================================================

----
bu metni markdown formatında nasıl kod bloğu içine alırım
```

---

### 4. Multimodal Füzyon ve FAISS Tabanlı Hibrit Kategori Tahmini

Klasör: `kategori_tahmini_2`

Bu proje, ürün görselleri ve ürün başlıklarından elde edilen embedding’leri birlikte kullanarak multimodal kategori tahmini yapar. Görsel ve metin temsilleri fuse edilerek daha güçlü bir ortak temsil oluşturulur. Model tahminleri ayrıca FAISS tabanlı yakın komşu sinyalleriyle desteklenir.

Amaç, yalnızca parametrik model tahminlerine değil, eğitim setindeki benzer ürünlerin dağılımına da bakarak daha güvenilir kategori tahminleri üretmektir.

Kullanılan başlıca teknolojiler:

- CLIP tabanlı embedding’ler
- PyTorch
- Multimodal fusion
- Attention mekanizmaları
- FAISS
- Cosine similarity

Örnek çıktı:

<img width="1919" height="435" alt="ornek_output" src="https://github.com/user-attachments/assets/db4ef3f2-545e-45bb-85d7-7acd092fa71c" />


Örnek iyileştirme özeti:

<img width="664" height="152" alt="ornek_iyilestirme_ozeti" src="https://github.com/user-attachments/assets/6bdc53ac-6f9e-4201-bf4e-ec77fb7932e2" />

---

### 5. OpenCV + CLAHE Ön İşleme ve Marqo-CLIP + LoRA Hiyerarşik Sınıflandırma

Klasör: `kategori_tahmini_3`

Bu proje, görsel ön işleme ve hiyerarşik kategori sınıflandırma adımlarını birlikte içerir. İlk aşamada OpenCV ve CLAHE kullanılarak görseller standart hale getirilir ve kontrast iyileştirmesi yapılır. İkinci aşamada Marqo-CLIP tabanlı görsel encoder üzerine LoRA katmanları eklenerek hiyerarşik kategori tahmini gerçekleştirilir.

Bu klasördeki iki ana dosya farklı aşamaları temsil eder:

- `opencv_clahe.py`: Görsel ön işleme, boyutlandırma ve kontrast iyileştirme
- `marqo_multimodal.ipynb`: Marqo-CLIP + LoRA tabanlı hiyerarşik sınıflandırma modeli

Kullanılan başlıca teknolojiler:

- OpenCV
- CLAHE
- Marqo-CLIP
- OpenCLIP
- LoRA
- PyTorch
- Hierarchical cascade head

Örnek test sonuçları:

<img width="1300" height="864" alt="ornek_test_sonucu_1" src="https://github.com/user-attachments/assets/03e60fdf-3dad-4eae-8029-0d8e2b4b976e" />

<img width="1297" height="1242" alt="ornek_test_sonucu_2" src="https://github.com/user-attachments/assets/0c381f2c-e031-48ca-b0b5-2ff8538f29fb" />

## Genel Amaç

Bu çalışmaların ortak amacı, e-ticaret platformlarında ürün görsellerini daha verimli işleyebilen bir yapay zekâ pipeline’ı oluşturmaktır. Projeler; ürünün görselde otomatik bulunması, gereksiz arka planlardan ayrıştırılması, görsel ve metin temsillerinin çıkarılması ve ürünlerin doğru kategori hiyerarşisine atanması gibi görevleri kapsar.

## Kullanılan Genel Teknolojiler

- Python
- PyTorch
- Hugging Face Transformers
- OpenCLIP
- ResNet50
- Florence-2
- GroundingDINO
- Marqo-CLIP
- LoRA
- FAISS
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Notlar

Kodlarda yer alan yerel dosya yolları çalıştırma ortamına göre güncellenmelidir. Özellikle Windows veya Linux kullanıcı dizinlerine ait sabit path değerleri, kendi sisteminizdeki veri ve çıktı klasörlerine göre değiştirilmelidir.

## Önerilen Kullanım

Her proje klasöründe önce ilgili `readme.md` dosyasının okunması önerilir. Notebook dosyaları Jupyter Lab veya Jupyter Notebook ile açılabilir. Python scriptleri ise gerekli bağımlılıklar kurulduktan sonra terminal üzerinden çalıştırılabilir.

```bash
python dosya_adi.py
```

Notebook çalıştırmak için:

```bash
jupyter lab
```
