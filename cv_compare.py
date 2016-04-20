from scipy.stats import chisquare
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def main():
	path = 'paintings/'
	orb = cv2.ORB(1000, 1.2)
	#dictionary for holding the histograms	
	index = {}
	#dictionary for holding the RGB images
	images = {}
	#dictionary for holding the bag of visual words information
	bagdex = {}
	bagdex_kp = {}
	#dictionary for holding greyscale images
	greyges = {}
	for filename in os.listdir(path):
		if filename.endswith('.jpg'):
			print filename
			image = cv2.imread(path + filename)
			
			#histogram data
			images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			hist = cv2.normalize(hist).flatten()
			index[filename] = hist
			
			#bovw data
			image = cv2.imread(path + filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
			greyges[filename] = image
			orb = cv2.ORB(1000, 1.2)
			kp, des = orb.detectAndCompute(greyges[filename], None)
			bagdex[filename] = des
			bagdex_kp[filename] = kp
	
	active = True
	while (active):
		query_filepath = raw_input("Please enter filepath of query image: \n")
		
		ind = query_filepath.find('/')
		query_filename = query_filepath
		if ind != -1:
			query_filename = query_filepath[ind+1:]
		
		# Create histogram for query image
		query_image = cv2.imread(query_filepath)
		images[query_filename] = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
		query_hist = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		query_hist = cv2.normalize(query_hist).flatten()
		index[query_filename] = query_hist
		bagsults = {}		

		# Create ORB feature descriptor for query image
		query_bagage = cv2.imread(query_filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		greyges[query_filename] = query_bagage
		query_kp, query_des = orb.detectAndCompute(greyges[query_filename], None)
		bagdex[query_filename] = query_des
		bagdex_kp[query_filename] = query_kp
		results = {}
		 
		# initialize the results dictionary for histogram
		methodName = "Hellinger"
		method = cv2.cv.CV_COMP_BHATTACHARYYA
		results = {}

		for (k, hist) in index.items():
			# compute the distance between the two histograms
			# using the method and update the results dictionary
			d = cv2.compareHist(index[query_filename], hist, method)
			results[k] = d

		# initialize the results dictionary for bovw
		FLANN_INDEX_LSH = 6
		index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2		
		search_params = dict(checks=50)

		for (k, des) in bagdex.items():
			d = 0
			"""
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			matches  = flann.knnMatch(index[query_filename], des, k=2)
			numMatches = 1
			totalDistance = 0
			matchesMask = [[0,0] for i in xrange(len(matches))]
			

			for i in range(0,500):
				if len(matches[i]) == 2:
					m = matches[i][0].distance
					n = matches[i][1].distance
					if m < 0.7 * n:
						matchesMask[i] = [1, 0]
						numMatches += 1

			draw_params = dict(matchColor = (0, 255, 0),
												singlePointColor = (255, 0, 0),
												matchesMask = matchesMask,
												flags = 0)
			img3 = drawMatches(query_image, query_kp, images[query_filename], query_kp, matches)
			plt.imshow(img3,),plt.show()			
			"""
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
			matches = bf.match(bagdex[query_filename], des)
			matches = sorted(matches, key=lambda val: val.distance)
			#out = drawMatches(query_image, query_kp, images[k], index_kp[k], matches[:10])
			for match in matches[:10]:
				d += match.distance
			bagsults[k] = d
	 
		# normalize the results
		hist_sum = 0
		bag_sum = 0
		for (name, value) in results.items():
			hist_sum += results[name]
			bag_sum += bagsults[name]
		for (name, value) in results.items():
			results[name] = value / float(hist_sum)
			bagsults[name] = bagsults[name] / float(bag_sum)
	 
	 	total_results = {}
	 	for (name, value) in results.items():
	 		total_results[name] = value + bagsults[name]
	 
		# sort the results
		total_results = sorted([(v, k) for (k, v) in total_results.items()], reverse = False)	
		results = sorted([(v, k) for (k, v) in results.items()], reverse = False)	
		bagsults = sorted([(v, k) for (k, v) in bagsults.items()], reverse = False)	

		# initialize the results figure
		fig = plt.figure("Results: ")

		# loop over the results
		n = 0
		for (i, (v, k)) in enumerate(bagsults):
			if n > 9:
				break
			n += 1
			# show the result
			ax = fig.add_subplot(1, 10, i + 1)
			ax.set_title("%s: %.2f" % (k, v))
			plt.imshow(images[k])
			plt.axis("off")
			
		#downloads the image into our own database
		#copyfile(query_filepath, 'paintings/copy_' + query_filename)
		 
		# show the OpenCV methods
		plt.show()




if __name__ == '__main__':
	main()
	
def drawMatches(img1, kp1, img2, kp2, matches):
    """
    Implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    implementation by rayryeng on stackoverflow
    http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        x1 = kp1[img1_idx].pt[0]
        y1 = kp1[img1_idx].pt[1]
        
        x2 = kp2[img2_idx].pt[0]
        y2 = kp2[img2_idx].pt[1]

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    fig = plt.figure("matches: ")
    plt.imshow(out)
   # cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')
    plt.show()

    # Also return the image if you'd like a copy
    return out
