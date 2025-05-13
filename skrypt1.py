import cv2
import numpy as np
import os
from glob import glob

input_folder = "Smashed"
output_folder = "czyste_zdjecia"
os.makedirs(output_folder, exist_ok=True)

def wykryj_kolory(hsv_image, gray_image):
    kolory = {
        'czerwony1': ((0, 70, 50), (10, 255, 255)),
        'czerwony2': ((170, 70, 50), (180, 255, 255)),
        'zielony':   ((36, 50, 50), (89, 255, 255)),
        'niebieski': ((90, 50, 50), (130, 255, 255)),
        'pomarańczowy': ((10, 100, 100), (25, 255, 255)),
        'fioletowy': ((130, 50, 50), (160, 255, 255)),
        'żółty': ((25, 50, 50), (35, 255, 255))
    }

    maska_koncowa = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for zakres in kolory.values():
        maska = cv2.inRange(hsv_image, np.array(zakres[0]), np.array(zakres[1]))
        maska_koncowa = cv2.bitwise_or(maska_koncowa, maska)
    
    _, black_mask = cv2.threshold(gray_image, 25, 255, cv2.THRESH_BINARY_INV)
    
    maska_koncowa = cv2.bitwise_or(maska_koncowa, black_mask)
    
    return maska_koncowa

def usun_napisy_na_brzegach(img):
    wysokosc, szerokosc = img.shape[:2]
    maska_napisow = np.zeros((wysokosc, szerokosc), dtype=np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    lewy_margines = int(0.10 * szerokosc)
    lewy_obszar = gray[:, 0:lewy_margines]
    
    _, lewy_mask = cv2.threshold(lewy_obszar, 200, 255, cv2.THRESH_BINARY)
    
    if np.sum(lewy_mask) > 0:
        kernel = np.ones((3, 3), np.uint8)
        lewy_mask = cv2.morphologyEx(lewy_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(lewy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:
                x, y, w, h = cv2.boundingRect(contour)
                
                if h > w and h < 100:
                    padding = 5
                    maska_napisow[max(0, y-padding):min(wysokosc, y+h+padding), 
                            0:min(szerokosc, x+w+padding)] = 255

    prawy_margines = int(0.10 * szerokosc)
    prawy_obszar = gray[:, szerokosc-prawy_margines:szerokosc]
    
    _, prawy_mask = cv2.threshold(prawy_obszar, 200, 255, cv2.THRESH_BINARY)
    
    if np.sum(prawy_mask) > 0:
        kernel = np.ones((3, 3), np.uint8)
        prawy_mask = cv2.morphologyEx(prawy_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(prawy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:
                x, y, w, h = cv2.boundingRect(contour)
                
                if h > w and h < 100:
                    padding = 5
                    maska_napisow[max(0, y-padding):min(wysokosc, y+h+padding), 
                            max(0, szerokosc-prawy_margines+x-padding):szerokosc] = 255
   
    gorny_margines = int(0.08 * wysokosc)
    gorny_obszar = gray[0:gorny_margines, :]

    _, gorny_mask = cv2.threshold(gorny_obszar, 200, 255, cv2.THRESH_BINARY)
    
    if np.sum(gorny_mask) > 0:
        kernel = np.ones((3, 3), np.uint8)
        gorny_mask = cv2.morphologyEx(gorny_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(gorny_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:
                x, y, w, h = cv2.boundingRect(contour)
                

                if w > h and w > 20:
                    padding = 5
                    maska_napisow[0:min(wysokosc, y+h+padding*2), 
                                 max(0, x-padding):min(szerokosc, x+w+padding)] = 255
    

    side_margin = int(0.05 * szerokosc)
    
    left_side = gray[:, 0:side_margin]
    bright_pixels_left = (left_side > 180).astype(np.uint8) * 255
    
    if np.sum(bright_pixels_left) > 100:
        maska_napisow[:, 0:side_margin] = bright_pixels_left
    
    right_side = gray[:, szerokosc-side_margin:szerokosc]
    bright_pixels_right = (right_side > 180).astype(np.uint8) * 255
    
    if np.sum(bright_pixels_right) > 100:
        maska_napisow[:, szerokosc-side_margin:szerokosc] = bright_pixels_right

    top_margin = int(0.05 * wysokosc)
    top_area = gray[0:top_margin, :]
    bright_pixels_top = (top_area > 180).astype(np.uint8) * 255
    
    if np.sum(bright_pixels_top) > 100:
        maska_napisow[0:top_margin, :] = bright_pixels_top
    
    return maska_napisow

pliki = glob(os.path.join(input_folder, "*.[jp][pn]g"))

for sciezka in pliki:
    nazwa = os.path.basename(sciezka)
    print(f"Przetwarzanie: {nazwa}")

    img_oryginalny = cv2.imread(sciezka)

    wysokosc, szerokosc = img_oryginalny.shape[:2]
    top = int(0.04 * wysokosc)
    bottom = int(0.96 * wysokosc)
    left = int(0.07 * szerokosc)
    right = int(0.93 * szerokosc)
    
    img = img_oryginalny[top:bottom, left:right]

    maska_napisow = usun_napisy_na_brzegach(img)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    maska_koncowa = wykryj_kolory(hsv, gray)
    maska_koncowa = cv2.bitwise_or(maska_koncowa, maska_napisow)
    
    kernel = np.ones((3, 3), np.uint8)
    maska_koncowa = cv2.dilate(maska_koncowa, kernel, iterations=1)
    
    wynik = cv2.inpaint(img, maska_koncowa, 3, cv2.INPAINT_TELEA)
    
    cv2.imwrite(os.path.join(output_folder, nazwa), wynik)

print("Zakończono.")
