

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.signal import convolve2d

USE_RANSAC = True
USE_BUILT_IN = False
USE_MY_IMAGES = True

def stitch_images(img1,img2,h1):

    s=np.dot(h,np.array([[*img1.shape[1::-1],1],[1,1,1],[img1.shape[1],1,1],[1,img1.shape[0],1]]).transpose())
    s=s[0:2,:]/s[2,:]
    s=np.concatenate((s,np.array([[0,img1.shape[1]],[0,img1.shape[0]]])),axis=1)

    smax=np.max(s,axis=1)
    smin=np.min(s,axis=1)

    fs=tuple(np.uint16(smax-smin))
    #fs=(fs[1],fs[0])

    h2=np.eye(3)
    if np.sum(smin)<0:
        ht=np.array([[1,0,-smin[0]],[0,1,-smin[1]],[0,0,1]])
        h1=np.dot(ht,h1)
        h2=ht

    normalizer=cv2.warpPerspective(np.ones_like(img1),h1,fs)+cv2.warpPerspective(np.ones_like(img2),h2,fs)
    normalizer[normalizer==0]=1
    im_out=(np.float32(cv2.warpPerspective(img1,h1,fs))+np.float32(cv2.warpPerspective(img2,h2,fs)))/np.float32(normalizer)
    im_out=np.uint8(im_out)
    return im_out

# Using this for comparision
def opencvFindHomography(match_xy):
    src_pts = match_xy[:, 0:2]
    dst_pts = match_xy[:, 2:4]
    print(src_pts)
    matrix, mask = cv2.findHomography(srcPoints=src_pts, dstPoints=dst_pts, method=cv2.RANSAC)
    return matrix

# implement this function
def myFindHomography(match_xy):
    # Find the points
    src_pts = match_xy[:, 0:2]
    dst_pts = match_xy[:, 2:4]

    # Set up a list for the A matrix. Ah = 0
    A_list = []

    ### Find the matrix rows of A
    for i in range(len(match_xy[:, 0])):
        x1 = src_pts[i, 0]
        y1 = src_pts[i, 1]
        x2 = dst_pts[i, 0]
        y2 = dst_pts[i, 1]
        ax = np.array([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        ay = np.array([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])

        A_list.append(ax)
        A_list.append(ay)

    ### Form A
    A = np.array(A_list)

    ### Solve for h
    # SVD
    u, s, vh = np.linalg.svd(A)

    # Find minimum singular value and corresponding vector
    s = list(s)
    min_val = min(s, key=abs)
    min_val_idx = s.index(min_val)
    h = vh[min_val_idx]

    ### Convert solution into matrix H
    H = np.reshape(h, (3,3))

    return H


# implement this function (Extra credit: +50%)
def myRANSAC(match_xy,K, EPS=900):
    # Split the points
    src_pts = match_xy[:, 0:2]
    dst_pts = match_xy[:, 2:4]

    # Find best score
    best_score = 0
    best_h = None
    best_inliers = None

    for i in range(K):
        # Choose 4 random points from the feature point sets
        idxs = np.random.random_integers(0, high = len(match_xy[:,0])-1, size=4)

        # Create the feature point set
        features = np.array([list(match_xy[i,:]) for i in idxs])

        # Find the homography
        h = myFindHomography(features)

        # Count inliers
        num_inliers = 0
        inliers = []
        for i in range(len(match_xy[:, 0])):
            p1 = [src_pts[i, 0], src_pts[i, 1], 0]
            p2 = [dst_pts[i, 0], dst_pts[i, 1], 0]

            # Find distance
            diff = p2 - h.dot(p1)
            dist = np.sqrt(diff.dot(diff))
            if dist < EPS:
                num_inliers += 1
                inliers.append([p1[0], p1[1], p2[0], p2[1]])

        # Check to see if this score should be kept
        if num_inliers > best_score:
            best_inliers = np.array(inliers)
            best_score = num_inliers
            best_h = h

    # Give back the best value
    return myFindHomography(best_inliers)




if __name__ == "__main__":
   # %matplotlib inline

    window_name='camera'

    if USE_MY_IMAGES:
        img1 = cv2.imread('personal1.jpg')
        img2 = cv2.imread('personal2.jpg')
    else:
        img1 = cv2.imread('IMAG4688.jpg')
        img2 = cv2.imread('IMAG4689.jpg')

    desc = cv2.xfeatures2d.SURF_create()

    ratio_thresh = 0.6

    kps1, descs1 = desc.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), None)
    kps2, descs2 = desc.detectAndCompute(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY), None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descs1, descs2, 2)

    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(len(good_matches))
    match_xy=np.array([[*(kps1[q.queryIdx].pt),*(kps2[q.trainIdx].pt)] for q in good_matches])

    # match_xy is a matrix with each row equals x1,y1,x2,y2,
    # where (x1,y1) and (x2,y2) are matched coordinates in img1 and img2, respectively
    # implement this function
    if USE_BUILT_IN:
        h = opencvFindHomography(match_xy)
    elif USE_RANSAC:
        EPS = 900 if USE_MY_IMAGES is not True else 2200
        h = myRANSAC(match_xy, 50, EPS)
    else:
        h = myFindHomography(match_xy)

    plt.figure(figsize=(15,15))
    plt.imshow(stitch_images(img1,img2,h))

    plt.show()