import cv2
import numpy as np
 
w = 500
h = 400
    
cap = cv2.VideoCapture(0)    

cap.set(3, w)
cap.set(4, h)

# "Frame" will get the next frame in the camera (via "cap"). 
# "Ret" will obtain return value from getting the camera frame, either true of false.
    
if cap.isOpened():
    ret, frame = cap.read()   # take first frame of the video
else:
    ret = False

while ret:
    
    ret, frame = cap.read()
    
    Z = frame.reshape((-1,3)) # 3 col and -1 means unknown rows,it figure out by itself
    Z = np.float32(Z)
    
    # criteria : It is the iteration termination criteria. 
    # When this criteria is satisfied, algorithm iteration stops. 
    # Actually, it should be a tuple of 3 parameters. They are ( type, max_iter, epsilon ):
    # a - type of termination criteria : It has 3 flags as below:
    # cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, 
    #                         is reached. cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm 
    #                         after the specified number of iterations, max_iter. cv2.TERM_CRITERIA_EPS
    #                         + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of 
    #                         he above condition is met.
    # b - max_iter - An integer specifying maximum number of iterations.
    # c - epsilon - Required accuracy


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    

    # samples (Z) : It should be of np.float32 data type, and each 
    #               feature should be put in a single column.
    # nclusters(K) : Number of clusters required at end
    # criteria : It is the iteration termination criteria. When this criteria is satisfied, 
    #           algorithm iteration stops. 
    # attempts : Flag to specify the number of times the algorithm is executed using 
    #            different initial labellings. The algorithm returns the labels that 
    #            yield the best compactness. This compactness is returned as output.
    # flags : This flag is used to specify how initial centers are taken. 
    #         Normally two flags are used for this : cv2.KMEANS_PP_CENTERS and cv2.KMEANS_RANDOM_CENTERS.

    K = 5
    ret, label1, center1 = cv2.kmeans(Z, K,None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Output parameters
    #   compactness : It is the sum of squared distance from each point to their corresponding centers.
    #   labels : This is the label array where each element marked 0,1,2....
    #   centers : This is array of centers of clusters.
    
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output1 = res1.reshape((frame.shape))
    
    cv2.imshow("Original", frame)
    cv2.imshow("Quantized", output1)
    if cv2.waitKey(1) == ord('q'): # exit on pressing q
        break

cv2.destroyAllWindows() # simply destroys all the windows we created
cap.release()           
