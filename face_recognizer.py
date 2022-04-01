import cv2

# enter user name same as it's trained model in .yml format
name = input("enter your name : ")

#calling haar face classifier
face_classifier = cv2.CascadeClassifier('hcascade.xml')

#this function will detect the face from the live cam and return the original and cropped image
def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)


model = cv2.face.LBPHFaceRecognizer_create()

# Loading trained model of the user  into the initialized model
model.read(f"{name}.yml")

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame) #will return back the actual img and cropped face 
    
    # to do exceptional handling
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        #print(results) -> shows co-ordinates of the image

        #to calculate confidence
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + f'% Confident it is {name}'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        #if confident enough
        if confidence > 85: #85%+ confidence
            cv2.putText(image, f"Hey {name}", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )

        #if still a doubt      
        elif confidence > 70: #70%+ confidence
            cv2.putText(image, "sit properly", (230, 450) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            cv2.imshow('Face Recognition', image )

        else: # if low confidence
            cv2.putText(image, "Locked : unknown user", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    #in case no face is detected
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key to exit
        break
        
cap.release()
cv2.destroyAllWindows()   