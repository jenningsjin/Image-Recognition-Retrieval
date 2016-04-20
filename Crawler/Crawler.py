import urllib
import re
import sys
import os

def main(argv):
	os.makedirs(argv[1])
	seedFile = open('seed.txt', 'r')
	urlText = seedFile.read().split(',')

	urlRegex = "https?:\/\/.+\.jpg"
	urlList = re.match(urlRegex, str(urlText))
	counter = 0
	for elt in urlText:
		searchObj = re.search(urlRegex, elt, flags=0)

		if searchObj is not None:
			# url = "http://uploads3.wikiart.org/images/caravaggio/the-death-of-the-virgin-1603(1).jpg"
			url = searchObj.group()

			outpath = argv[1] + '/' + argv[1] +str(counter) + ".jpg"
			image = urllib.URLopener()
			image.retrieve(url, outpath)
			print counter
			counter+=1

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "you need <PROGRAM NAME> <ROOTFILENAME>" 
	else:
		main(sys.argv)