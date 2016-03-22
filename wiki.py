import pandas as pd
import vislab
import vislab.datasets
import os
import subprocess


df = pd.read_csv('wikipaintings_urls.csv')

#vislab.dataset.download_images(df,os.getcwd())
FNULL = open(os.devnull, 'w')
i = 0

for name, url in zip(df['image_id'],df['image_url']):
	name = name+".jpg"
	path = "original/"
	subprocess.call(["wget", url,"-O", path+name],stdout=FNULL, stderr=subprocess.STDOUT)

	if i %1000 ==0:
		print i
	i+=1



print "hellow?"
