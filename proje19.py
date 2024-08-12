


import cv2
import numpy as np

 
def region_of_interest(img, vertices):

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=(0, 0, 139), thickness=10):

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def process_image(image, counted_cars):

    # Görüntüyü gri tonlamalı hale getirme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gürültüyü azaltma ve görüntüyü yumuşatma
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenarları tespit etme
    edges = cv2.Canny(blur, 50, 150)
    
    # ROI (Region of Interest) belirleme
    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = [
        (50, height),
        (width / 2, height / 2 + 50),
        (width - 50, height)
    ]

    cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), (184, 134, 11))
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Dönüşümü kullanarak çizgileri bulma
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
    

    # Çizgileri görüntüye çizme
    line_image = np.zeros_like(image)
    if lines is not None:
        draw_lines(line_image, lines, color=(184, 134, 11), thickness=10)  # Kalınlığı artırma
        
    # Nesne tanıma modelini yükleme
    car_cascade = cv2.CascadeClassifier('C:\\Users\\EXCALIBUR\\Desktop\\haarcascade_car.xml')
    

    # Kaskad sınıflandırıcısını başarıyla yükleme kontrolü
    if car_cascade.empty():
        print("Error: Cascade classifier not loaded!")
        return image, 0
    

    # Görüntüyü gri tona dönüştürme ve nesne tespiti
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))  # minSize parametresini (50, 50) olarak ayarladık
    

    # Her bir aracı daire ile çizme ve sayma
    for (x, y, w, h) in cars:

        # Daireyi çiz
        cv2.circle(image, (x + w//2, y + h//2), max(w, h)//2, (184, 134, 11), 3)  # Kalınlığı artır

        # Dairenin merkez koordinatlarını bul
        center = (x + w//2, y + h//2)

        # Dairenin merkez koordinatlarını kullanarak, diğer dairelerle çakışıp çakışmadığını kontrol et
        is_new_car = True

        for (counted_center, _, _) in counted_cars:
            # Eğer mevcut daire bir önceki daireyle çakışıyorsa, yeni araç olarak kabul etme

            if np.linalg.norm(np.array(center) - np.array(counted_center)) < max(w, h) * 1.5:  # Dairelerin merkezlerinin uzaklığı, yarıçapın 1.5 katından küçükse çakışma kabul edilir
                is_new_car = False
                break
        # Eğer mevcut daire çakışmıyorsa, yeni araç olarak kabul et ve say
        if is_new_car:
            counted_cars.append((center, w, h))
    
    # Görüntüyü birleştirme
    processed_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    
    return processed_image, counted_cars



# Video dosyasını yükleme
video = cv2.VideoCapture('görüntü.mp4.mp4')



total_car_count = 0
counted_cars = []

while True:

    ret, frame = video.read()
    if not ret:
        break
    
    processed_frame, counted_cars = process_image(frame, counted_cars)  # process_image fonksiyonunu çağır ve araç sayısını al
    
    total_car_count = len(counted_cars)  # Toplam araç sayısını güncelle
    
    # Görüntüyü biraz daha büyük göstermek için boyutunu artır
    processed_frame = cv2.resize(processed_frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow('Video', processed_frame)
    
    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Toplam araç sayısını yazdır
print("Total detected cars:", total_car_count)

video.release()
cv2.destroyAllWindows()





# Şerit değişikliği algılamak için eşik değeri
threshold = 10

def detect_lane_change(prev_left_lane, prev_right_lane, left_lane, right_lane):
    if prev_left_lane is not None and prev_right_lane is not None:
        # Önceki ve mevcut şerit pozisyonlarını karşılaştır
        if abs(prev_left_lane - left_lane) > threshold or abs(prev_right_lane - right_lane) > threshold:
            print("Şerit değişikliği algılandı!")
            return True
    return False

# İlk video karesini alır ve kenarları algılamak için gri tonlamaya dönüştürür
video = cv2.VideoCapture('görüntü.mp4')

# Önceki çerçevedeki şeritlerin pozisyonunu saklamak için değişkenler
prev_left_lane = None
prev_right_lane = None

# Bayrak tanımla
lane_changed = False

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Bulanıklaştırma (Bulanıklık Kaldırma)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Renk Uzayı Dönüşümü (HSV)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Sarı ve Beyaz Renklerin Algılanması (Renk Maskesi Oluşturma)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)  # Siyahlık oranını azaltmak için eşik değerini 180'e düşürdüm
    upper_white = np.array([255, 50, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    mask = cv2.bitwise_or(yellow_mask, white_mask)

    # Morfolojik İşlemler (Gürültü Giderme)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Şerit pozisyonlarını bulma
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_lane = None
    right_lane = None
    if contours:
        # En büyük konturu bul
        biggest_contour = max(contours, key=cv2.contourArea)
        # Konturun sınırlayıcı kutusunu al
        x, y, w, h = cv2.boundingRect(biggest_contour)
        # Şerit merkezini hesapla
        left_lane = x + w // 4  # Sol şerit merkezi
        right_lane = x + 3 * w // 4  # Sağ şerit merkezi

    # Şerit değişikliği algılamak için fonksiyonu çağırma
    if not lane_changed:
        lane_changed = detect_lane_change(prev_left_lane, prev_right_lane, left_lane, right_lane)

    # Önceki şerit pozisyonlarını güncelle
    prev_left_lane = left_lane
    prev_right_lane = right_lane

    # Çıktıyı göster
    cv2.imshow('Frame', mask)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma ve pencereleri kapatma
video.release()
cv2.destroyAllWindows()
