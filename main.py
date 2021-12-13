import cv2
import mediapipe as mp
#from mediapipe.python.solutions.hands import Hands

mp_draw = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands

video = cv2.VideoCapture(0) #0 means primarey camera

with mp_hand.Hands(min_detection_confidence = 0.5,
                min_tracking_confidence = 0.5) as hands:
    while True:
        ret,image = video.read() #if webcam ok ret will be 1 else 0
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #converting img to rgb bcz .process only prcess rgb
        image.flags.writeable = False #holding that image
        results = hands.process(image) #processing the image 
        image.flags.writeable = True #let the image free
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #bk it again in bgr
        
        if results.multi_hand_landmarks : #if there is any or more hand_landmark it will be true
            for hand_landmark in results.multi_hand_landmarks: #if true then it will iterate all the hand marks & draw dots & lines over them
                mp_draw.draw_landmarks(image,hand_landmark,mp_hand.HAND_CONNECTIONS)
        
        
        cv2.imshow("frame",image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

video.release()
cv2.destroyAllWindows()