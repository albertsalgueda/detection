# Import opencv
import cv2 

# Import uuid
import uuid

# Import Operating System
import os

# Import time
import time

labels = ['water', 'cash', 'cup', 'fork']
number_imgs = 20

IMAGES_PATH = os.path.join('assets/images')
LABELIMG_PATH = os.path.join('assets/labelImg')
"""
if not os.path.exists(IMAGES_PATH):
    raise Exception('path not found')
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir(path)

for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        # Flip the frame
        frame = cv2.flip(frame, 1)
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

#LABEL THE IMAGES
"""
if not os.path.exists(LABELIMG_PATH):
    os.mkdir(LABELIMG_PATH)
    os.system("git clone https://github.com/tzutalin/labelImg")

os.system('cd labelImg && pipenv run make qt5py3 && pipenv run python3 labelImg.py')


#"""