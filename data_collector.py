import cv2
import os

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('hcascade.xml')

#enter the name of user whose data is being collected
NAME = input("enter your name : ")


# This function will detect faces and return that cropped face else a nonetype value is returned
def face_extractor(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert RGB img to gray
    faces = face_classifier.detectMultiScale(gray, 1.3, 5) 
    
    if faces is (): #if no face is detected
        return None
    
    # if a face detected, crop it down and return it
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w] 

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)

count = 0
os.mkdir(f"./{NAME}")

# This will collect 200 samples of user's face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None: # if a face is detected
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = f'./{NAME}/{NAME}' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count to track the no current clicks out of 200
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else: #if no face is found - in bg (termial prints)
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count >= 200: #13 is the Enter Key to quit
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete ...")