# Face Authentication System for Online Exam

## About the Project

*An intriguing Python project based not only on face detection but face recognization using OpenCV, Image Processing and Local Binary Patterns Histogram (LBPH).
This Project I re-built during the time college  was encouraging us students to learn and work upon Machine Learning in groups. *

### Instructions

**Pre-requisites**

- python3 installed
- opencv installed `pip install opencv-python`
- numpy installed `pip install numpy`
- haar cascade file present ("hcascade.xml present in repo)

**1. data_collector.py**
*This program will detect a human face and only crop the required part, convert to black-white format and save as the dataset for training the mode*
    `python3 data_collector.py`

**2. model_trainer.py**
*This program will train a ML model and save it using OpenCV and the training dataset collected from the above program*
    `python3 model_trainer.py`

**3. face_recognizer.py**
*This program will detect the face, use the trained model to recognize the user while telling its confidence level of recognition*
    `python3 face_recognizer.py`

***NOTE*** - *This project and files are developed under fedora environment, running it under another env such as windows might give failure errors due to different filesystem and path format.*
