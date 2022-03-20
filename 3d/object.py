import cv2
import mediapipe as mp
import time

#DOC: https://google.github.io/mediapipe/solutions/models.html#objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

#logo = cv2.imread("assets/logo.png",0)
#size = 100
#gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
#logo = cv2.resize(gray,(size,size),interpolation =cv2.INTER_CUBIC)
#_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

with mp_objectron.Objectron(static_image_mode = False,
                            max_num_objects =2,
                            min_detection_confidence = 0.5,
                            min_tracking_confidence = 0.8,
                            model_name = 'Shoe') as objectron:                        
    while cap.isOpened():

        # Read the a frame from webcam, initialize time 
        success,frame = cap.read()
        start = time.time() #we need it to calculate time complexity of the program

        #resize the image in case that our processing power is low
        frame = cv2.resize(frame,(640,480))

        """
        #add the FPS to the image
        text = "FPS" + str(int(1/(time.time()-last_time)))
        last_time = time.time()
        cv2.putText(frame,text,(10,38),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        """
        
        #Change image color's type
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #To improve performance, mark the image as not writable to
        frame.flags.writeable = False

        #We call the objectron's method ".process" and pass current frame 
        results = objectron.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #Check if we detect any object & writting  
        if results.detected_objects:
            for detected_object in results.detected_objects:
                 mp_drawing.draw_landmarks(frame,detected_object.landmarks_2d,mp_objectron.BOX_CONNECTIONS) #convert predicted to 2d
                 mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

        #Caculate time to record and process 1 frame
        end = time.time() 
        totalTime = end - start
        fps = 1/totalTime
        
        #flip the image so it feels like a mirror
        frame = cv2.flip(frame,1)

        #insert data to cam and show
        cv2.putText(frame,f'FPS: {int(fps)}', (20,20), cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)

        # Show the frame in a window
        cv2.imshow("cam",frame)

        # Check if q has been pressed to quit
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
