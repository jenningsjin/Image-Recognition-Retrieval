Image Recognition for EECS 498

run: python cv_compare.py

The program will read in all the images in the 'paintings/' folder.
It will ask for a input of the full relative path with filename: ex. 'paintings/Impressionism4.jpg'
The program will then output ten images (including the query image) that are most similar to the query (based on both color histogram and ORB feature descriptors)
Close out of the results window that appears, and the program will prompt for input again

folder hierarchy

-Image-Recognition-Retrieval
	-paintings/
		-HighRen1.jpg
		-HighRen2.jpg
		...
	-Crawler/
		-Crawler.py
		-seed.txt
		-readme
	-results/
		-samples of results from queries
	-compare.py
	-README.md
