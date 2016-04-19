from scipy.stats import chisquare
import matplotlib.pyplot as plt
import cv2
import os

def main():
	path = 'paintings/'
	index = {}
	images = {}
	for filename in os.listdir(path):
		if filename.endswith('.jpg'):
			print filename
			image = cv2.imread(path + filename)
			images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			
			hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			hist = cv2.normalize(hist).flatten()
			index[filename] = hist
			
		# METHOD #1: UTILIZING OPENCV
	# initialize OpenCV methods for histogram comparison
	#hellinger works most as expected
	OPENCV_METHODS = (
		("Correlation", cv2.cv.CV_COMP_CORREL),
		("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
		("Intersection", cv2.cv.CV_COMP_INTERSECT), 
		("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))
	 
	# loop over the comparison methods
	for (methodName, method) in OPENCV_METHODS:
		# initialize the results dictionary and the sort
		# direction
		results = {}
		reverse = False
	 
		# if we are using the correlation or intersection
		# method, then sort the results in reverse order
		if methodName in ("Correlation", "Intersection"):
			reverse = True

		for (k, hist) in index.items():
			# compute the distance between the two histograms
			# using the method and update the results dictionary
			d = cv2.compareHist(index["land1.jpg"], hist, method)
			results[k] = d
	 
		# sort the results
		results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)	

		# show the query image
		fig = plt.figure("Query")
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(images["land1.jpg"])
		plt.axis("off")
	 
		# initialize the results figure
		fig = plt.figure("Results: %s" % (methodName))
		fig.suptitle(methodName, fontsize = 20)
	 
		# loop over the results
		for (i, (v, k)) in enumerate(results):
			# show the result
			ax = fig.add_subplot(1, len(images), i + 1)
			ax.set_title("%s: %.2f" % (k, v))
			plt.imshow(images[k])
			plt.axis("off")
	 
	# show the OpenCV methods
	plt.show()




if __name__ == '__main__':
	main()
