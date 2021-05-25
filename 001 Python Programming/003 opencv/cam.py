from imutils import face_utils
import dlib
import cv2
import numpy as np
 
"""  
task1
"""
image = cv2.imread("person.jpg", cv2.IMREAD_COLOR)
sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR) # RGB
sunglasses_trans = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED) # 투명도때문에 필요

height, width, channel = image.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), -90, height/width)
dst = cv2.warpAffine(image, matrix, (width, height))

resized = cv2.resize(dst, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

while True:
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        print(rect.top(), rect.bottom(), rect.left(), rect.right())
        cropped_img = resized[rect.top():rect.bottom(), rect.left():rect.right()]
        # cv2.imshow("crop", cropped_img)

        # 간격 구하고 ratio로 선글라스 크기 조절
        temp = np.array(shape)
        min_x = min(temp[:,0])
        max_x = max(temp[:,0])
        sun_width = max_x - min_x
        ratio = float(sun_width) / sunglasses.shape[1]
        sun_height = int(sunglasses.shape[0] * ratio)
        dim = (sun_width, sun_height)
        resized_sunglasses = cv2.resize(sunglasses, dim, interpolation=cv2.INTER_AREA)
        resized_sunglasses_trans = cv2.resize(sunglasses_trans, dim, interpolation=cv2.INTER_AREA)
        
        # for(x, y) in shape:
        #     cv2.circle(resized, (x, y), 2, (0, 255, 0), -1)

    for i in range(sun_height):
        for j in range(sun_width):
            if resized_sunglasses_trans[i,j,3] == 0:
                pass
            else:
                cropped_img[ 20+i, 8+j, :] = resized_sunglasses[i,j,:]

    cv2.imshow("output", resized)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()


""" 
task2
"""
cap = cv2.VideoCapture(0)
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    try:
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            cropped_img = image[rect.top():rect.bottom(), rect.left():rect.right()]

            detect = detector(cropped_img, 0)
            shape1 = predictor(cropped_img, detect[0])

            temp = np.array(shape)
            min_x = min(temp[:,0])
            max_x = max(temp[:,0])
            sun_width = max_x - min_x
            ratio = float(sun_width) / sunglasses.shape[1]
            sun_height = int(sunglasses.shape[0] * ratio)
            dim = (sun_width, sun_height)
            resized_sunglasses = cv2.resize(sunglasses, dim, interpolation=cv2.INTER_AREA)
            resized_sunglasses_trans = cv2.resize(sunglasses_trans, dim, interpolation=cv2.INTER_AREA)

            left_x = shape1.part(0).x
            y1 = shape1.part(38).y
            y2 = shape1.part(19).y
            up = int((3*y1+y2)/4)
            
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        for i in range(sun_height):
            for j in range(sun_width):
                if resized_sunglasses_trans[i,j,3] == 0:
                    pass
                else:
                    cropped_img[ up+i, left_x+j, :] = resized_sunglasses[i,j,:]
    except:
        pass
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()