import numpy as np
import argparse
import time
import cv2
import os

# Türkçe çeviri sözlüğü
TURKCE_ISIMLER = {
    "person": "kişi",
    "bicycle": "bisiklet",
    "car": "araba",
    "motorbike": "motosiklet",
    "aeroplane": "uçak",
    "bus": "otobüs",
    "train": "tren",
    "truck": "kamyon",
    "boat": "tekne",
    "traffic light": "trafik lambası",
    "fire hydrant": "yangın musluğu",
    "stop sign": "dur işareti",
    "parking meter": "parkmetre",
    "bench": "bank",
    "bird": "kuş",
    "cat": "kedi",
    "dog": "köpek",
    "horse": "at",
    "sheep": "koyun",
    "cow": "inek",
    "elephant": "fil",
    "bear": "ayı",
    "zebra": "zebra",
    "giraffe": "zürafa",
    "backpack": "sırt çantası",
    "umbrella": "şemsiye",
    "handbag": "el çantası",
    "tie": "kravat",
    "suitcase": "bavul",
    "frisbee": "frizbi",
    "skis": "kayak",
    "snowboard": "snowboard",
    "sports ball": "spor topu",
    "kite": "uçurtma",
    "baseball bat": "beyzbol sopası",
    "baseball glove": "beyzbol eldiveni",
    "skateboard": "kaykay",
    "surfboard": "sörf tahtası",
    "tennis racket": "tenis raketi",
    "bottle": "şişe",
    "wine glass": "şarap bardağı",
    "cup": "fincan",
    "fork": "çatal",
    "knife": "bıçak",
    "spoon": "kaşık",
    "bowl": "kase",
    "banana": "muz",
    "apple": "elma",
    "sandwich": "sandviç",
    "orange": "portakal",
    "broccoli": "brokoli",
    "carrot": "havuç",
    "hot dog": "sosisli",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "pasta",
    "chair": "sandalye",
    "sofa": "kanepe",
    "pottedplant": "saksı bitkisi",
    "bed": "yatak",
    "diningtable": "yemek masası",
    "toilet": "tuvalet",
    "tvmonitor": "televizyon",
    "laptop": "laptop",
    "mouse": "fare",
    "remote": "kumanda",
    "keyboard": "klavye",
    "cell phone": "cep telefonu",
    "microwave": "mikrodalga",
    "oven": "fırın",
    "toaster": "tost makinesi",
    "sink": "lavabo",
    "refrigerator": "buzdolabı",
    "book": "kitap",
    "clock": "saat",
    "vase": "vazo",
    "scissors": "makas",
    "teddy bear": "oyuncak ayı",
    "hair drier": "saç kurutma makinesi",
    "toothbrush": "diş fırçası"
}

# Görüntüyü nesne tespiti için uygun boyuta getir
def goruntuyu_uygun_hale_getir(img, max_width=1920, max_height=1080, min_width=416, min_height=416):
    """
    Görüntüyü nesne tespiti için uygun boyuta getirir.
    Çok büyük görüntüleri küçültür, çok küçük görüntüleri büyütür.
    En-boy oranını korur.
    """
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Çok büyük görüntüleri küçült
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Görüntü küçültüldü: {width}x{height} -> {new_width}x{new_height}")
    
    # Çok küçük görüntüleri büyüt
    elif width < min_width or height < min_height:
        scale = max(min_width / width, min_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        print(f"Görüntü büyütüldü: {width}x{height} -> {new_width}x{new_height}")
    
    return img, original_size

# YOLO modelini yükle
def yukle_yolo():
    # Dosya yolları
    weights_path = "yolov3 (1).weights"
    config_path = "yolov3.cfg.txt"
    names_path = "coco.names.txt"
    
    # Dosyaların varlığını kontrol et
    if not os.path.exists(weights_path):
        print(f"Hata: {weights_path} dosyası bulunamadı!")
        return None, None, None
    if not os.path.exists(config_path):
        print(f"Hata: {config_path} dosyası bulunamadı!")
        return None, None, None
    if not os.path.exists(names_path):
        print(f"Hata: {names_path} dosyası bulunamadı!")
        return None, None, None
    
    # YOLO network'ünü yükle
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Sınıf isimlerini yükle
    with open(names_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Output layer'ları al
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

# Nesne tespiti yap
def nesne_tespiti_yap(img, net, output_layers):
    height, width, channels = img.shape
    
    # Görüntüyü blob formatına çevir (YOLO için)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    return outputs, width, height

# Tespit sonuçlarını işle
def tespit_sonuclarini_isle(outputs, width, height, confidence_threshold=0.5, nms_threshold=0.4):
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Nesne koordinatlarını hesapla
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Sol üst köşe koordinatları
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-maximum suppression uygula (çakışan kutuları temizle)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    return boxes, confidences, class_ids, indices

# Görüntüyü çiz ve etiketle
def goruntuyu_ciz(img, boxes, confidences, class_ids, indices, classes):
    # Rastgele renkler oluştur (her sınıf için)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Türkçe çeviri varsa kullan
            turkce_label = TURKCE_ISIMLER.get(label, label)
            
            # Renk seç
            color = colors[class_ids[i]]
            
            # Dikdörtgen çiz
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Etiket metni (nesne adı + güven skoru)
            text = f"{turkce_label} {confidence:.2f}"
            
            # Metin boyutunu hesapla
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Metin arka planı için dikdörtgen çiz
            cv2.rectangle(img, (x, y - text_height - baseline - 5), 
                         (x + text_width, y), color, -1)
            
            # Metni yaz
            cv2.putText(img, text, (x, y - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img

# Resim dosyalarını bul
def resim_dosyalarini_bul():
    """Klasördeki resim dosyalarını bulur"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    # Mevcut klasördeki tüm dosyaları listele
    for filename in os.listdir('.'):
        if os.path.isfile(filename):
            # Dosya uzantısını kontrol et
            _, ext = os.path.splitext(filename.lower())
            if ext in image_extensions:
                image_files.append(filename)
    
    return image_files

# Ana fonksiyon - Resim dosyasından nesne tespiti
def main():
    # Argümanları parse et
    parser = argparse.ArgumentParser(description='YOLO Nesne Tespiti - Resim Dosyası')
    parser.add_argument('--resim', type=str, default=None,
                       help='İşlenecek resim dosyası yolu (opsiyonel)')
    parser.add_argument('--tumu', action='store_true',
                       help='Tüm resimleri işle')
    args = parser.parse_args()
    
    # YOLO modelini yükle
    print("YOLO modeli yükleniyor...")
    start_time = time.time()
    net, classes, output_layers = yukle_yolo()
    
    if net is None:
        print("Model yüklenemedi! Program sonlandırılıyor.")
        return
    
    load_time = time.time() - start_time
    print(f"Model yüklendi! (Yükleme süresi: {load_time:.2f} saniye)")
    
    # Resim dosyalarını belirle
    if args.resim:
        # Komut satırından belirtilen resim
        if os.path.exists(args.resim):
            selected_images = [args.resim]
        else:
            print(f"Hata: {args.resim} dosyası bulunamadı!")
            return
    elif args.tumu:
        # Tüm resimleri işle
        selected_images = resim_dosyalarini_bul()
        if not selected_images:
            print("\nKlasörde resim dosyası bulunamadı!")
            print("Lütfen klasöre .jpg, .jpeg, .png, .bmp veya .gif formatında resim ekleyin.")
            return
    else:
        # Kullanıcıdan resim seçimi
        image_files = resim_dosyalarini_bul()
        
        if not image_files:
            print("\nKlasörde resim dosyası bulunamadı!")
            print("Lütfen klasöre .jpg, .jpeg, .png, .bmp veya .gif formatında resim ekleyin.")
            return
        
        print(f"\nBulunan resim dosyaları: {len(image_files)}")
        for i, img_file in enumerate(image_files, 1):
            print(f"{i}. {img_file}")
        
        # Kullanıcıdan resim seçimi
        print("\n" + "="*50)
        print("RESİM SEÇİMİ")
        print("="*50)
        print("1. Tek bir resim seçmek için numarasını girin")
        print("2. Tüm resimleri işlemek için 'tümü' yazın")
        print("3. Çıkmak için 'q' yazın")
        print("="*50)
        
        choice = input("\nSeçiminiz: ").strip().lower()
        
        if choice == 'q':
            print("Program sonlandırıldı.")
            return
        elif choice == 'tümü' or choice == 'tumu' or choice == 'all':
            selected_images = image_files
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(image_files):
                    selected_images = [image_files[index]]
                else:
                    print("Geçersiz seçim! Program sonlandırıldı.")
                    return
            except ValueError:
                print("Geçersiz seçim! Program sonlandırıldı.")
                return
    
    # Seçilen resimleri işle
    for img_path in selected_images:
        print(f"\nİşleniyor: {img_path}")
        
        # Resmi yükle
        img = cv2.imread("idris.jpg")
        if img is None:
            print(f"Hata: {img_path} yüklenemedi!")
            continue
        
        # Görüntüyü nesne tespiti için uygun boyuta getir
        img, original_size = goruntuyu_uygun_hale_getir(img)
        
        # Nesne tespiti yap
        print("Nesne tespiti yapılıyor...")
        tespit_baslangic = time.time()
        outputs, width, height = nesne_tespiti_yap(img, net, output_layers)
        
        # Sonuçları işle
        boxes, confidences, class_ids, indices = tespit_sonuclarini_isle(outputs, width, height)
        tespit_suresi = time.time() - tespit_baslangic
        
        # Görüntüyü çiz ve etiketle
        img = goruntuyu_ciz(img, boxes, confidences, class_ids, indices, classes)
        
        # Tespit edilen nesneleri göster
        print(f"Tespit edilen nesne sayısı: {len(indices) if len(indices) > 0 else 0}")
        print(f"Tespit süresi: {tespit_suresi:.2f} saniye")
        if len(indices) > 0:
            for i in indices.flatten():
                label = str(classes[class_ids[i]])
                turkce_label = TURKCE_ISIMLER.get(label, label)
                print(f"  - {turkce_label}: {confidences[i]:.2%}")
        
        # Sonuç görüntüsünü kaydet
        output_path = f"tespit_sonucu_{os.path.basename(img_path)}"
        cv2.imwrite(output_path, img)
        print(f"Sonuç kaydedildi: {output_path}")
        
        # Görüntüyü göster
        cv2.imshow(f"Nesne Tespiti - {img_path} (Çıkmak için 'q' tuşuna basın)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\nİşlem tamamlandı!")

if __name__ == "__main__":
    main()
