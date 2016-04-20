from scipy.stats import chisquare
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def main():
	path = 'paintings/'
		
	index = {}
	images = {}
	for filename in os.listdir(path):
		if filename.endswith('.jpg'):
			print filename
			image = cv2.imread(path + filename)
			images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#images[filename] = ""
			#cv2.cvtColor(image, images[filename], cv2.COLOR_BGR2GRAY)
			

			detector = cv2.FeatureDetector_create("ORB")
			kp = detector.detect(images[filename])
			extractor = cv2.DescriptorExtractor_create("ORB")
			kp, des = extractor.compute(images[filename], kp)
			
			
			index[filename] = des
			#print des
	
	

	active = True
	while (active):
		#query_filepath = raw_input("Please enter filepath of query image: \n")
		#query_filepath = "paintings/HighRen22.jpg"
		query_filepath = "paintings/starry_night.jpg"
		active = False
		ind = query_filepath.index('/')
		query_filename = query_filepath
		if ind != -1:
			query_filename = query_filepath[ind+1:]
		
		# Create ORB feature descriptor for query image
		query_image = cv2.imread(query_filepath)
		#cv2.cvtColor(query_image, images[query_filename], cv2.COLOR_BGR2GRAY)
		images[query_filename] = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
		
		detector = cv2.FeatureDetector_create("ORB")
		query_kp = detector.detect(images[filename])
		extractor = cv2.DescriptorExtractor_create("ORB")
		query_kp, query_des = extractor.compute(images[filename], query_kp)
		
		index[query_filename] = query_des
		 

		results = {}
		#FLANN_INDEX_KDTREE = 0
		#index_params= dict(algorithm = FLANN_INDEX_KDTREE,
    #               table_number = 12, # 12
    #               key_size = 20,     # 20
    #               multi_probe_level = 2) #2		
		#search_params = dict(checks=50)

		for (k, des) in index.items():
			# compute the distance between the two histograms
			# using the method and update the results dictionary
			
			#flann = cv2.FlannBasedMatcher(index_params, search_params)
			#matches  = flann.knnMatch(index[query_filename], des, k=2)
			#numMatches = 0
#			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	#		matches = bf.knnMatch(index[query_filename], des, k=2)
			
			#matches = sorted(matches, key = lambda x:x.distance)
			

#			for m,n in enumerate(matches):
#				if m.distance < 0.7*n.distance:
#					numMatches += 1
			
			print k
			d = np.linalg.norm(np.subtract(index[query_filename], des))
			#d = numMatches
			print d
			#d = cv2.compareHist(index[query_filename], hist, method)
			
			
			results[k] = d
	 
		# sort the results
		results = sorted([(v, k) for (k, v) in results.items()], reverse = False)	
	 
		# initialize the results figure
		fig = plt.figure("Results: ")
	 
		# loop over the results
		#n = 0
		for (i, (v, k)) in enumerate(results):
		#	if n > 3:
			#	break
			#n += 1
			# show the result
			ax = fig.add_subplot(1, len(images), i + 1)
			ax.set_title("%s: %.2f" % (k, v))
			plt.imshow(images[k])
			plt.axis("off")
	 
	# show the OpenCV methods
	plt.show()




if __name__ == '__main__':
	main()
