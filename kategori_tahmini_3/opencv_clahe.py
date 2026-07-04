import cv2
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import threading
import queue
import os

class OpenCVGPUCLAHEProcessor:
    def __init__(self):
        self.min_clip_limit = 0.5
        self.max_clip_limit = 4.0
        self.min_tile_size = 4
        self.max_tile_size = 16
        
        # OpenCV GPU desteği kontrolü
        self.gpu_available = self.check_opencv_gpu()
        
    def check_opencv_gpu(self):
        """OpenCV GPU desteğini kontrol et"""
        try:
            # CUDA device sayısını kontrol et
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0:
                print(f"🚀 OpenCV CUDA desteği tespit edildi! {gpu_count} GPU bulundu")
                
                # GPU memory bilgisi
                device_info = cv2.cuda.DeviceInfo(0)
                total_memory = device_info.totalMemory() / (1024**3)
                print(f"💾 GPU Memory: {total_memory:.1f} GB")
                return True
            else:
                print("⚠️  OpenCV CUDA desteği bulunamadı")
                return False
        except Exception as e:
            print(f"⚠️  OpenCV GPU kontrolü başarısız: {e}")
            return False
    
    def gpu_analyze_image(self, img):
        """OpenCV GPU ile görüntü analizi"""
        if not self.gpu_available:
            return self.cpu_analyze_image(img)
        
        try:
            # GPU'ya yükle
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # GPU'da grayscale'e çevir
            gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            
            # GPU'dan CPU'ya indir (histogram için)
            gray = gpu_gray.download()
            
            # İstatistikler (vectorized)
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            min_val, max_val = gray.min(), gray.max()
            
            # Histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_normalized = hist / hist.sum()
            
            # Entropi
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            
            # GPU'da edge detection
            gpu_edges = cv2.cuda.Canny(gpu_gray, 50, 150)
            edges = gpu_edges.download()
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Histogram peaks
            hist_peaks = self._fast_peak_count(hist_normalized)
            
            return {
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness,
                'contrast': std_brightness / mean_brightness if mean_brightness > 0 else 0,
                'entropy': entropy,
                'edge_density': edge_density,
                'dynamic_range': max_val - min_val,
                'histogram_peaks': hist_peaks,
            }
            
        except Exception as e:
            print(f"GPU analiz hatası, CPU'ya geçiliyor: {e}")
            return self.cpu_analyze_image(img)
    
    def cpu_analyze_image(self, img):
        """CPU fallback analizi"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # İstatistikler
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_val, max_val = gray.min(), gray.max()
        
        # Histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_normalized = hist / hist.sum()
        
        # Entropi
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Histogram peaks
        hist_peaks = self._fast_peak_count(hist_normalized)
        
        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'contrast': std_brightness / mean_brightness if mean_brightness > 0 else 0,
            'entropy': entropy,
            'edge_density': edge_density,
            'dynamic_range': max_val - min_val,
            'histogram_peaks': hist_peaks,
        }
    
    def _fast_peak_count(self, hist, threshold=0.001):
        """Hızlı peak counting"""
        significant = hist > threshold
        peaks = 0
        in_peak = False
        
        for is_sig in significant:
            if is_sig and not in_peak:
                peaks += 1
                in_peak = True
            elif not is_sig:
                in_peak = False
        
        return min(peaks, 10)
    
    def calculate_optimal_clip_limit(self, properties):
        """Dinamik clip_limit hesaplama"""
        factors = []
        
        # Kontrast faktörü
        contrast_factor = np.clip(1.0 / (properties['contrast'] + 0.1), 0.5, 3.0)
        factors.append(contrast_factor)
        
        # Entropi faktörü
        entropy_factor = np.clip(properties['entropy'] / 8.0, 0.5, 2.5)
        factors.append(entropy_factor)
        
        # Dinamik aralık faktörü
        range_factor = np.clip(255.0 / (properties['dynamic_range'] + 1), 0.8, 3.0)
        factors.append(range_factor)
        
        # Parlaklık faktörü
        if properties['mean_brightness'] < 60:
            brightness_factor = 3.0
        elif properties['mean_brightness'] > 200:
            brightness_factor = 1.0
        else:
            x = (properties['mean_brightness'] - 128) / 64
            brightness_factor = 2.5 - 1.5 / (1 + np.exp(-x))
        factors.append(brightness_factor)
        
        # Ağırlıklı ortalama
        weights = [0.3, 0.25, 0.25, 0.2]
        clip_limit = np.average(factors, weights=weights)
        
        return np.clip(clip_limit, self.min_clip_limit, self.max_clip_limit)
    
    def calculate_optimal_tile_size(self, properties, img_shape):
        """Dinamik tile_size hesaplama"""
        height, width = img_shape[:2]
        
        # Görüntü boyutu faktörü
        size_factor = min(height, width) / 512.0
        
        # Kenar yoğunluğu faktörü
        edge_factor = properties['edge_density'] * 20
        
        # Histogram peaks faktörü
        peak_factor = min(properties['histogram_peaks'] / 5.0, 1.0)
        
        # Base tile size
        base_tile_size = 8 * size_factor
        
        # Detay ayarlaması
        if edge_factor > 1.0 or peak_factor > 0.6:
            tile_size = base_tile_size * 1.5
        else:
            tile_size = base_tile_size
        
        # Sınırları kontrol et
        tile_size = max(self.min_tile_size, min(self.max_tile_size, tile_size))
        tile_size = int(tile_size // 2 * 2)  # Çift sayı yap
        
        return (tile_size, tile_size)
    
    def get_optimal_parameters(self, img):
        """Görüntü için optimal parametreleri hesapla"""
        properties = self.gpu_analyze_image(img)
        
        clip_limit = self.calculate_optimal_clip_limit(properties)
        tile_size = self.calculate_optimal_tile_size(properties, img.shape)
        
        return {
            'clip_limit': clip_limit,
            'tile_grid_size': tile_size,
        }
    
    def gpu_clahe_apply(self, img, clip_limit, tile_grid_size):
        """OpenCV GPU ile CLAHE uygulama"""
        if not self.gpu_available:
            return self.cpu_clahe_apply(img, clip_limit, tile_grid_size)
        
        try:
            # LAB color space'e çevir
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # GPU'ya yükle
            gpu_l = cv2.cuda_GpuMat()
            gpu_l.upload(l)
            
            # GPU CLAHE oluştur ve uygula
            gpu_clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            gpu_l_processed = gpu_clahe.apply(gpu_l)
            
            # CPU'ya indir
            l_processed = gpu_l_processed.download()
            
            # Kanalları birleştir
            lab_processed = cv2.merge([l_processed, a, b])
            result = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)
            
            return result
            
        except Exception as e:
            print(f"GPU CLAHE hatası, CPU'ya geçiliyor: {e}")
            return self.cpu_clahe_apply(img, clip_limit, tile_grid_size)
    
    def cpu_clahe_apply(self, img, clip_limit, tile_grid_size):
        """CPU CLAHE fallback"""
        # LAB color space'e çevir
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CPU CLAHE uygula
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_processed = clahe.apply(l)
        
        # Kanalları birleştir
        lab_processed = cv2.merge([l_processed, a, b])
        result = cv2.cvtColor(lab_processed, cv2.COLOR_LAB2BGR)
        
        return result

class BatchProcessor:
    def __init__(self, input_dir, output_dir, max_workers=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.processor = OpenCVGPUCLAHEProcessor()
        
        print(f"🔧 Ayarlar: {self.max_workers} worker")
        print(f"🚀 GPU Desteği: {'Aktif' if self.processor.gpu_available else 'Devre Dışı'}")
    
    def get_image_files(self):
        """Desteklenen görsel dosyalarını listele"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.jfif'}
        
        image_files = []
        for ext in extensions:
            # Hem küçük hem büyük harf için arama
            image_files.extend(list(self.input_dir.rglob(f'*{ext}')))
            image_files.extend(list(self.input_dir.rglob(f'*{ext.upper()}')))
        
        # Duplicate'leri kaldır
        image_files = list(set(image_files))
        
        return sorted(image_files)
    
    def process_single_image(self, input_path):
        """Tek görsel işleme"""
        try:
            # Görüntüyü yükle
            img = cv2.imread(str(input_path))
            if img is None:
                return {'status': 'failed', 'error': 'Görüntü yüklenemedi', 'path': input_path}
            
            # Optimal parametreleri hesapla
            params = self.processor.get_optimal_parameters(img)
            
            # CLAHE uygula (GPU veya CPU)
            result = self.processor.gpu_clahe_apply(
                img, 
                params['clip_limit'], 
                params['tile_grid_size']
            )
            
            # Çıktı dosya yolunu oluştur
            relative_path = input_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path
            
            # Klasör yapısını oluştur
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Kaydet
            success = cv2.imwrite(str(output_path), result)
            
            if success:
                return {
                    'status': 'success', 
                    'path': input_path,
                    'output_path': output_path,
                    'params': params
                }
            else:
                return {'status': 'failed', 'error': 'Kaydetme hatası', 'path': input_path}
                
        except Exception as e:
            return {'status': 'failed', 'error': str(e), 'path': input_path}
    
    def format_time(self, seconds):
        """Süreyi formatla"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def calculate_eta(self, processed, total, elapsed_time):
        """Kalan süreyi hesapla"""
        if processed == 0:
            return "Hesaplanıyor..."
        
        avg_time_per_image = elapsed_time / processed
        remaining_images = total - processed
        eta_seconds = remaining_images * avg_time_per_image
        
        return self.format_time(eta_seconds)
    
    def process_batch(self):
        """Batch işleme"""
        print("🔍 Görsel dosyalar taranıyor...")
        image_files = self.get_image_files()
        total_files = len(image_files)
        
        if total_files == 0:
            print("❌ Hiç görsel dosya bulunamadı!")
            return
        
        print(f"✅ {total_files:,} görsel dosya bulundu")
        print(f"📁 Kaynak: {self.input_dir}")
        print(f"📁 Hedef: {self.output_dir}")
        print("-" * 60)
        
        # Çıktı klasörünü oluştur
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Başlangıç zamanı
        start_time = time.time()
        
        # Progress bar ve istatistik değişkenleri
        successful_count = 0
        failed_count = 0
        total_clip_limit = 0
        total_tile_size = 0
        
        # Progress bar ile paralel işleme
        with tqdm(total=total_files, 
                 desc="🎨 OpenCV GPU CLAHE", 
                 unit="görsel",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Tüm görevleri submit et
                future_to_file = {
                    executor.submit(self.process_single_image, img_file): img_file 
                    for img_file in image_files
                }
                
                # Sonuçları işle
                for future in as_completed(future_to_file):
                    result = future.result()
                    
                    # İstatistikleri güncelle
                    if result['status'] == 'success':
                        successful_count += 1
                        if 'params' in result:
                            total_clip_limit += result['params']['clip_limit']
                            total_tile_size += result['params']['tile_grid_size'][0]
                    else:
                        failed_count += 1
                        # İlk 5 hatayı göster
                        if failed_count <= 5:
                            print(f"\n⚠️  Hata: {result['path'].name} - {result['error']}")
                    
                    # Progress bar güncelle
                    processed_total = successful_count + failed_count
                    elapsed_time = time.time() - start_time
                    eta = self.calculate_eta(processed_total, total_files, elapsed_time)
                    
                    # Progress bar description güncelle
                    gpu_status = "GPU" if self.processor.gpu_available else "CPU"
                    pbar.set_description(
                        f"🎨 {gpu_status} CLAHE | ✅{successful_count} ❌{failed_count} | ETA: {eta}"
                    )
                    pbar.update(1)
        
        # İşlem tamamlandı
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 OPENCV GPU CLAHE İŞLEME TAMAMLANDI!")
        print("="*60)
        print(f"📊 Toplam İşlenen: {total_files:,} görsel")
        print(f"✅ Başarılı: {successful_count:,} ({successful_count/total_files*100:.1f}%)")
        print(f"❌ Başarısız: {failed_count:,} ({failed_count/total_files*100:.1f}%)")
        print(f"⏱️  Toplam Süre: {self.format_time(total_time)}")
        print(f"⚡ Ortalama Hız: {total_files/total_time:.1f} görsel/saniye")
        print(f"🚀 GPU Desteği: {'Kullanıldı' if self.processor.gpu_available else 'CPU Fallback'}")
        
        if successful_count > 0:
            avg_clip = total_clip_limit / successful_count
            avg_tile = total_tile_size / successful_count
            print(f"📈 Ortalama Clip Limit: {avg_clip:.2f}")
            print(f"📐 Ortalama Tile Size: {avg_tile:.1f}")
        
        print(f"💾 Çıktı Klasörü: {self.output_dir}")
        print("="*60)

def main():
    """Ana fonksiyon"""
    print("🚀 OPENCV GPU CLAHE İŞLEMCİSİ")
    print("="*60)
    
    # Klasör yolları
    input_directory = r"C:\Users\selam\Desktop\images"
    output_directory = r"C:\Users\selam\Desktop\images_clahe"
    
    # Optimized ayarlar
    max_workers = max(1, multiprocessing.cpu_count())
    
    try:
        # Processor oluştur ve çalıştır
        processor = BatchProcessor(
            input_dir=input_directory,
            output_dir=output_directory,
            max_workers=max_workers
        )
        
        # İşlemi başlat
        processor.process_batch()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu!")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()