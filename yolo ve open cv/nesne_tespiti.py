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
    
    # YOLO network'ünü yükle
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Sınıf isimlerini yükle
    with open(names_path, 'r') as f:
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
            
            # Renk seç
            color = colors[class_ids[i]]
            
            # Dikdörtgen çiz
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Etiket metni (nesne adı + güven skoru)
            text = f"{label} {confidence:.2f}"
            
            # Metin boyutunu hesapla
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Metin arka planı için dikdörtgen çiz
            cv2.rectangle(img, (x, y - text_height - baseline - 5), 
                         (x + text_width, y), color, -1)
            
            # Metni yaz
            cv2.putText(img, text, (x, y - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img

# Ana fonksiyon - Canlı video akışı
def main():
    # Argümanları parse et
    parser = argparse.ArgumentParser(description='YOLO Nesne Tespiti - IP Kamera')
    parser.add_argument('--ip', type=str, default=None,
                       help='Telefon IP adresi (örn: 192.168.1.100)')
    parser.add_argument('--port', type=str, default='8080',
                       help='Telefon port numarası (varsayılan: 8080)')
    args = parser.parse_args()
    
    # YOLO modelini yükle
    print("YOLO modeli yükleniyor...")
    net, classes, output_layers = yukle_yolo()
    print("Model yüklendi!")
    
    # IP adresini al
    if args.ip is None:
        print("\n" + "="*50)
        print("TELEFON KAMERASI BAĞLANTISI")
        print("="*50)
        print("1. Telefonunuza 'IP Webcam' uygulamasını yükleyin")
        print("2. Uygulamayı başlatın ve 'Start Server' butonuna basın")
        print("3. Ekranda gösterilen IP adresini aşağıya girin")
        print("="*50)
        ip_address = input("\nTelefon IP adresini girin (örn: 192.168.1.100): ").strip()
        if not ip_address:
            ip_address = "192.168.1.100"
            print(f"Varsayılan IP kullanılıyor: {ip_address}")
    else:
        ip_address = args.ip
    
    # Video URL'ini oluştur
    video_url = f"http://{ip_address}:{args.port}/video"
    print(f"\nTelefon kamerasına bağlanılıyor: {video_url}")
    cap = cv2.VideoCapture(video_url)
    
    # Video akışını kontrol et
    if not cap.isOpened():
        print(f"\nHata: '{video_url}' adresine bağlanılamadı!")
        print("\nKontrol edin:")
        print("1. Telefon ve bilgisayar aynı WiFi ağında mı?")
        print("2. IP Webcam uygulamasında 'Start Server' aktif mi?")
        print("3. IP adresi doğru mu? (IP Webcam ekranında gösterilen)")
        print("4. Firewall IP adresini engelliyor olabilir mi?")
        print(f"\nTekrar denemek için: python nesne_tespiti.py --ip {ip_address}")
        return
    
    print("Video akışı başlatıldı! Çıkmak için 'q' tuşuna basın.")
    
    # FPS hesaplama için
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Video akışı kesildi!")
                break
            
            # Görüntüyü nesne tespiti için uygun boyuta getir
            frame, _ = goruntuyu_uygun_hale_getir(frame, max_width=1280, max_height=720)
            
            # Nesne tespiti yap
            outputs, width, height = nesne_tespiti_yap(frame, net, output_layers)
            
            # Sonuçları işle
            boxes, confidences, class_ids, indices = tespit_sonuclarini_isle(outputs, width, height)
            
            # Görüntüyü çiz ve etiketle
            frame = goruntuyu_ciz(frame, boxes, confidences, class_ids, indices, classes)
            
            # FPS hesapla ve göster
            fps_frame_count += 1
            if fps_frame_count % 10 == 0:
                fps_end_time = time.time()
                fps = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            # FPS ve nesne sayısını göster
            cv2.putText(frame, f"FPS: {fps:.1f} | Nesneler: {len(indices) if len(indices) > 0 else 0}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Görüntüyü göster
            cv2.imshow("Canlı Nesne Tespiti - 'q' ile çıkış", frame)
            
            # 'q' tuşuna basıldığında çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nProgram durduruldu!")
    finally:
        # Temizlik
        cap.release()
        cv2.destroyAllWindows()
        print("Video akışı kapatıldı.")

if __name__ == "__main__":
    main()

