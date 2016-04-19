from scipy.stats import chisquare
import cv2
import os

def main():
	n = 3 #num_paintings
	histogram_features = dict(list())
	filename = 'histogram_features.txt'
	f = open(filename, 'r')
	for i in range(0,n):
		name = f.readline()
		#print name
		histogram = []
		for j in range(0,256):
			histogram.append(float(f.readline()))
		histogram_features[name] = histogram
		
	#print histogram_features['flower.jpg\n']
	#for key, value in histogram_features.items():
	#	print key
	print chisquare(histogram_features['flower.jpg\n'])
	print chisquare(histogram_features['Starry_Night_Over_the_Rhone.jpg\n'])
	
	
	
if __name__ == '__main__':
	main()
