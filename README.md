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

- `ornek_output.png`

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

- `ornek_output.png`

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

- `ornek_output.txt`

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

Örnek çıktılar:

- `ornek_output.png`
- `ornek_iyilestirme_ozeti.png`

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

Örnek çıktılar:

- `ornek_test_sonucu_1.png`
- `ornek_test_sonucu_2.png`

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
