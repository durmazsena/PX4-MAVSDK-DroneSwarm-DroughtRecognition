# PX4-MAVSDK-DroneSwarm-DroughtRecognition ✅

**Sena DURMAZ, Arda YILDIZ, Bahattin Eren YILDIRIM, Buse KÜÇÜKKÖMÜRCÜ, Türkay ÖZBEK**

---

## Genel Bakış
Bu proje, drone sürümleri (swarm) tarafından toplanan görüntüler ile LoveDA veri kümesini kullanarak 7 sınıflı (Background, Other, Building, Road, Water, Agriculture, Forest) semantic segmentation gerçekleştirmeyi amaçlar. Model mimarisi olarak Hugging Face üzerinden alınmış ve ADE fine-tuned ağırlıklarıyla initialize edilmiş **SegFormer-B3** tercih edilmiştir. Eğitim esnasında Exponential Moving Average (EMA), combined loss (Focal + Dice), ve çeşitli augmentasyon teknikleri kullanılmıştır.

---

## Hızlı Başlangıç
**Gereksinimler (örnek):**
- Python 3.8+
- torch, torchvision
- transformers
- albumentations, albumentations.pytorch
- numpy, Pillow, matplotlib, tqdm

Örnek kurulum:
```
pip install torch torchvision transformers albumentations matplotlib pillow tqdm
```

---

## Veri Kümesi ve Dizini
- Kullanılan veri: LoveDA
- Beklenen kök dizin örneği: `/content/drive/MyDrive/LoveDa-Dataset`
- Yapı:
  - `Train/Train/{Urban,Rural}/images_png` ve `masks_png`
  - `Val/Val/{Urban,Rural}/...`
  - `Test/Test/{Urban,Rural}/images_png`
- Maske özel durumları: `ignore_index = 255` (görmezden gelinir)
- Sınıf sayısı: 7 (0..6)

---

## Model & Eğitim Konfigürasyonu (Teknik Detaylar) 🔧
- Model: `SegformerForSemanticSegmentation` ("nvidia/segformer-b3-finetuned-ade-512-512")
- `config.num_labels = 7`
- `ignore_mismatched_sizes=True` ile kısmi yüklemeye izin veriliyor

Eğitim hiperparametreleri (kodda kullanılan):
- Görüntü boyutu: **512x512**
- Batch size: **8** (train), Test batch size: **2**
- Num workers: **2**
- Epochs: **10** (örnek) — öneri: daha yüksek epoch sayıları (30-50) ile deneyin
- Optimizer: **AdamW**, lr = **1e-4**
- Scheduler: **ReduceLROnPlateau**(mode='min', factor=0.5, patience=2)
- Early stopping patience: **3** (validation loss gelişmezse durdurulur)
- EMA: decay = **0.999** (doğrulamada EMA modeli kullanılır)
- Mixed precision: tavsiye edilir (AMP) — hız ve bellek kazanımı için

Augmentasyonlar (train):
- Resize(512,512)
- HorizontalFlip(p=0.5), VerticalFlip(p=0.3), RandomRotate90(p=0.5)
- ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
- RandomBrightnessContrast, ColorJitter
- Normalize(ImageNet mean/std) + ToTensorV2()

Validation/Test augment: Resize + Normalize + ToTensorV2

---

## Kayıp Fonksiyonları (Loss) 🧠
- **Focal Loss** (CrossEntropy tabanlı, gamma=2, ignore_index=255)
- **Dice Loss** (one-hot target, softmax on logits, ignore 255)
- **CombinedLoss** = 0.5 * Focal + 0.5 * Dice
- Örnek sınıf ağırlıkları (kodda): `tensor([0.5, 0.8, 1.5, 1.2, 2.2, 3.0, 1.0])`

---

## Eğitim Döngüsü & Checkpointing
- Eğitim sırasında model.train() ile ağırlıklar güncellenir. EMA modeli paralel olarak güncellenir ve doğrulama EMA ile yapılır.
- Validation loss iyileşirse **EMA modelinin ağırlıkları kaydedilir**: `best_segformer_b3_ema_.pth`
- Loss hesaplarında maske pikselleri `255` ise dışlanır; sınıflar `torch.clamp(..., max=6)` ile sınırlandırılır.

Öneriler:
- Reproducibility için seed sabitleyin (`torch.manual_seed`, numpy, vb.)
- Logging: TensorBoard / wandb ile metric ve görselleştirme kaydedin
- Multi-GPU veya DDP kullanacaksanız batch size ve gradient accumulation planlayın

---

## İnferans & Post-processing ✅
- Test-time augmentation: horizontal flip ensemble (orijinal+flip -> average logits)
- Mask post-processing: `refine_agriculture_class` fonksiyonu
  - Yerel 3x3 patch içinde `Agriculture (5)` yoğunluğu ≥ 3 ise çevredeki `Other (1)` pikselleri `Agriculture (5)` olarak düzeltilir
  - Küçük boyutta (örneğin 128x128) uygulanıp sonra nearest interpolation ile 512x512'ye ölçeklenir
- Renk haritası `label_colors` kullanılarak görselleştirme yapılır

---

## Swarm Entegrasyonu (PX4 + MAVSDK) 🚁
Aşağıda repo içindeki mevcut scriptlere (örn. `drone1.py`, `drone2.py`, `ucak1.py`, `ucak2.py`, `startall.sh`, `stop_all.sh`, `start_video_logger1.py`, `start_video_logger2.py`) dayanarak adım adım kullanım kılavuzu bulunmaktadır.

### Öne çıkan dosyalar
- `drone1.py`, `drone2.py` / `ucak1.py`, `ucak2.py`: Her bir drone için telemetry toplama, SHM (shared memory) yazma ve flocking (sürü davranışı) kontrolü içerir.
- `start_video_logger1.py`, `start_video_logger2.py`: Hareket algılanınca GPS bindirmeli video kaydı yapan yardımcı scriptler.
- `startall.sh`: QGroundControl, PX4 SITL (iki drone simülasyonu) ve drone scriptlerini başlatmaya yönelik yardımcı script.
- `stop_all.sh`: Sistem süreçlerini güvenli şekilde kapatır (PX4, mavsdk_server, loggerler vb.).
- Configuration: `drone1_config.ini`, `drone2_config.ini` (örnek içerik: `[swarm]\nID = 1\nConnection = udp://:14541\nPort = 50051`).

---

### Gereksinimler & Hazırlık
- PX4 Autopilot (SITL) kurulumu ve build edilmiş `px4` ikili dosyaları.
- QGroundControl (isteğe bağlı, uçuşu görselleştirmek için).
- `mavsdk` Python paketi ve `mavsdk_server` çalışır durumda.
- Gerekli Python paketleri (repo kökünde belirtilenler).

### Konfigürasyon
- `drone1_config.ini` ve `drone2_config.ini` içinde:
  - `ID`: Benzersiz drone kimliği (örn. 1, 2)
  - `Connection`: MAVLink bağlantı stringi (örn. `udp://:14541`)
  - `Port`: Scriptte kullanılan mavsdk portu (örn. 50051 / 50052)
- Scriptler config dosyasını varsayılan `~/Masaüstü/SP-494/*_config.ini` yolundan okur. Konfigürasyon konumunu değiştirecekseniz kodu güncelleyin veya sembolik link koyun.

### Telemetry ve Paylaşılan Bellek
- SHM adı: `telemetry_shared`, boyut: `4096` byte.
- Telemetry kolonları: latitude, longitude, absolute_altitude, speed, yaw, battery_percent, satellites_visible vb.
- Her drone telemetry verisini SHM'e yazar; flocking controller diğer dronların konumlarını okuyarak koordine hareket sağlar.

---

### Adım Adım Kullanım (Simülasyon için)
1. `startall.sh` ile QGroundControl ve iki PX4 SITL örneğini başlatın (script terminal başlıkları ve beklemeler içerir).
2. `startall.sh` ayrıca `drone1.py` / `drone2.py` (veya `ucak1.py`/`ucak2.py`) scriptlerini başlatır. Alternatif olarak manuel:
   - `python3 drone1.py` (veya `ucak1.py`)
   - `python3 drone2.py`
3. Hareket başladığında video kaydı için:
   - `python3 start_video_logger1.py`
   - `python3 start_video_logger2.py`
   Bu scriptler giriş olarak tanımlı bir video dosyasından oynatma yapıp GPS bindirmeli bir çıktı oluşturur (örnek yollar script içinde sabitlenmiştir, ihtiyaca göre güncelleyin).
4. Süreçleri durdurmak için: `./stop_all.sh` (tmux, px4, mavsdk_server ve logger süreçlerini sonlandırır).

---

### Güvenlik & Operasyonel Notlar
- Scriptler otomatik arming ve takeoff komutları gönderir; simülasyonda test edin, gerçek uçuşta manuel onay/safeguard ekleyin.
- Kodda arming için kısa beklemeler (4s) ve takeoff sonrası beklemeler (22–25s) bulunur; uçakların stabilize olması için önemlidir.
- SHM oluşturulurken çakışma durumları ele alınır (mevcut alan varsa bağlanılır). Script sonlanınca SHM temizleme (unlink) yapılırsa sadece oluşturan kapatır.
- `stop_all.sh` acil durum kapanışı sağlar; gerçek saha uçuşlarında ek güvenlik katmanları (kill switch, fail-safe) ekleyin.

---

### Veri Toplama & Model Entegrasyonu
- Video çıktıları GPS metadata ile kaydedilir (scriptler örnek sabit yollar kullanır). Gerçek görüntü kaydı için kamera stream'lerini kaydeden ufak bir logger ile entegre edin.
- Veri pipeline önerisi:
  1. Videoları frame'lere ayırın.
  2. Her frame için JSON metadata (drone_id, timestamp, lat, lon, alt) oluşturun.
  3. Annotasyon/maske oluşturma: El ile veya semi-otomatik araçlarla maskeleri hazırlayın ve LoveDA formatına uygun isimlendirme uygulayın.
- Online inference: modeli uçakta çalıştırmak yerine yer istasyonunda çalıştırmak daha pratiktir. Eğer uçakta çalıştırılacaksa modeli TorchScript/ONNX'e çevirip quantize ederek düşük-latency uygulama hazırlayın.

---

### Hata Ayıklama & Log'lar
- Terminallerdeki renkli çıktı (GREEN, YELLOW, RED vb.) ve `[SHM]` mesajları hızlı durum tespiti sağlar.
- MAVSDK bağlantı hataları, JSON parse hataları veya SHM hataları genellikle loglarda görünür; önce bu loglara bakın.

---

### Örnek Komutlar
- Simülasyonu başlat: `./startall.sh`
- Sadece drone scriptlerini çalıştırmak: `python3 drone1.py & python3 drone2.py &`
- Video kaydı başlat: `python3 start_video_logger1.py & python3 start_video_logger2.py &`
- Sistemi durdur: `./stop_all.sh`

---

Bu kılavuz README içine eklendi ve sonuna soru eklenmedi.

---

## Model ağırlığı paylaşımı
Depoda `best_segformer_b3_ema_.pth` dosyası yer almıyorsa model ağırlıkları paylaşılmamış demektir. Bu yüzden **eğitimi tekrar etmeniz** gerekli olacaktır. Aşağıda reproducible eğitim adımları bulunmaktadır.

### Tekrar Eğitim (reproducible) - örnek adımlar:
1. Veri dizinini hazırla (`/path/to/LoveDa-Dataset`)
2. Ortamı kur: gerekli paketleri yükle
3. Kodun `train_segformer` fonksiyonunu içeren bir script (`train.py`) oluştur veya mevcut script'i kullan
4. Örnek komut:
```
python train.py --data /path/to/LoveDa-Dataset --epochs 30 --batch-size 8 --lr 1e-4 --device cuda --amp --save-dir ./checkpoints
```
5. En iyi EMA checkpoint: `./checkpoints/best_segformer_b3_ema_.pth`
6. Değerlendirme / mIoU hesaplama için `val` setini kullanın ve sonuçları loglayın

---

## İpuçları & İleri Çalışmalar
- Model küçültme için: ONNX export → quantize (dynamic/static) → uçakta deploy
- Sliding-window veya tile-based inference ile yüksek çözünürlü görüntülerde segmentasyon yapın
- Daha agresif augmentasyonlarla (mixup, cutmix, gridmask) sınıf dengesizliğini azaltabilirsiniz
- Cross-validation ile daha güvenilir mIoU elde edebilirsiniz

---

## Dosya Yapısı (Öne Çıkan Dosyalar)
- `segment_and_detect_agriculture.py` — segmentasyon ve tarla düzeltme mantığı
- `ucak1.py`, `ucak2.py` — drone entegrasyon scriptleri
- `startall.sh`, `stop_all.sh` — çalışma scriptleri
- `model_and_test_data/` — yardımcı veriler ve test scriptleri

---