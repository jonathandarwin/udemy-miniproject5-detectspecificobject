import cv2
import numpy as np

def sift_detector(newImage, templateImage):
    image1 = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    image2 = templateImage

    sift = cv2.xfeatures2d.SIFT_create()

    # cari kp dan desc dari kedua gambar (crop image dan image yang ingin dicari)
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
    searchParams = dict(checks = 100)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(desc1, desc2, k=2)

    goodMatches = []
    for m,n in matches:
        # matches yang bagus merupakan match yang distance nya seperti rumus dibawah
        # tujuan ny untuk mengeliminasi / mengurangi match2 yang tidak tepat
        if (m.distance < 0.7 * n.distance):
            goodMatches.append(m)
    
    return len(goodMatches)

cap = cv2.VideoCapture(0)

templateImage = cv2.imread('box_in_scene.png', 0)

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # buat kotak 
    # kotak ini buat deteksi gambar yang dicari
    # kalo gambar yang dicari ada dikotak ini, munculkan pesan object found
    topLeftX = width/3
    topLeftY = (height/2) + (height/4)
    bottomRightX = (width/3) * 2
    bottomRightY = (height/2) - (height/4)

    cv2.rectangle(frame, (topLeftX, topLeftY), (bottomRightX, bottomRightY), 255, 3)

    # crop gambar kotak yang sudah dibuat diatas
    cropped = frame[bottomRightY:topLeftY, topLeftX:bottomRightX]

    # flip gambar nya biar videony kayak mirror
    frame = cv2.flip(frame, 1)

    matches = sift_detector(cropped, templateImage)

    cv2.putText(frame, str(matches), (450,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 1)

    threshold = 10

    if(matches > threshold):
        cv2.rectangle(frame, (topLeftX, topLeftY), (bottomRightX, bottomRightY), 255, 3)
        cv2.putText(frame, 'Object Found', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 1)
    cv2.imshow('Object Detector Using SIFT', frame)
    if(cv2.waitKey(1) == 13):
        break

cap.release()