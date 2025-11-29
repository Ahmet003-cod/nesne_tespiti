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
    
    # Çok büyük görüntüleri küçült
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Çok küçük görüntüleri büyüt
    elif width < min_width or height < min_height:
        scale = max(min_width / width, min_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return img

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

# Ana fonksiyon - Webcam'den canlı nesne tanıma
def main():
    # Argümanları parse et
    parser = argparse.ArgumentParser(description='YOLO Nesne Tanıma - Kamera')
    parser.add_argument('--kamera', type=int, default=0,
                       help='Kamera indeksi (varsayılan: 0)')
    parser.add_argument('--genislik', type=int, default=1280,
                       help='Kamera genişliği (varsayılan: 1280)')
    parser.add_argument('--yukseklik', type=int, default=720,
                       help='Kamera yüksekliği (varsayılan: 720)')
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
    
    # Kamerayı aç
    print(f"\nKamera {args.kamera} açılıyor...")
    cap = cv2.VideoCapture(args.kamera)
    
    # Kamera kontrolü
    if not cap.isOpened():
        print(f"Hata: Kamera {args.kamera} açılamadı!")
        print("\nKontrol edin:")
        print("1. Kamera bağlı mı?")
        print("2. Başka bir program kamera kullanıyor olabilir mi?")
        print("3. Farklı bir kamera numarası deneyin (1, 2, vb.)")
        print("   Örnek: python kamera_nesne_tanima.py --kamera 1")
        return
    
    # Kamera ayarları
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.genislik)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.yukseklik)
    
    print("Kamera başlatıldı! Çıkmak için 'q' tuşuna basın.")
    print("Fotoğraf çekmek için 's' tuşuna basın.")
    
    # FPS hesaplama için
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    frame_count = 0
    indices = []
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Kameradan görüntü alınamadı!")
                break
            
            # Görüntüyü nesne tespiti için uygun boyuta getir
            frame = goruntuyu_uygun_hale_getir(frame, max_width=args.genislik, max_height=args.yukseklik)
            
            # Her 3 karede bir nesne tespiti yap (performans için)
            if frame_count % 3 == 0:
                # Nesne tespiti yap
                outputs, width, height = nesne_tespiti_yap(frame, net, output_layers)
                
                # Sonuçları işle
                boxes, confidences, class_ids, indices = tespit_sonuclarini_isle(outputs, width, height)
                
                # Görüntüyü çiz ve etiketle
                frame = goruntuyu_ciz(frame, boxes, confidences, class_ids, indices, classes)
                
                # Tespit edilen nesneleri konsola yazdır (her 30 karede bir)
                if frame_count % 30 == 0 and len(indices) > 0:
                    detected_objects = {}
                    for i in indices.flatten():
                        label = str(classes[class_ids[i]])
                        turkce_label = TURKCE_ISIMLER.get(label, label)
                        if turkce_label not in detected_objects:
                            detected_objects[turkce_label] = 0
                        detected_objects[turkce_label] += 1
                    
                    if detected_objects:
                        print("Tespit edilen nesneler:", ", ".join([f"{k}: {v}" for k, v in detected_objects.items()]))
            
            frame_count += 1
            
            # FPS hesapla ve göster
            fps_frame_count += 1
            if fps_frame_count % 10 == 0:
                fps_end_time = time.time()
                fps = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            # FPS ve nesne sayısını göster
            nesne_sayisi = len(indices) if len(indices) > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f} | Nesneler: {nesne_sayisi}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Görüntüyü göster
            cv2.imshow("Canlı Nesne Tanıma - 'q' ile çıkış, 's' ile fotoğraf", frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Fotoğraf kaydet
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"kamera_fotografi_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Fotoğraf kaydedildi: {filename}")
                
    except KeyboardInterrupt:
        print("\nProgram durduruldu!")
    finally:
        # Temizlik
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera kapatıldı.")

if __name__ == "__main__":
    main()
