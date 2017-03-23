import PIL
from PIL import Image
import os
width = 224

imgFolder = 'caltech250Train/'
destFolder = 'resizedCaltech250Train/'

folders = os.listdir(imgFolder)
for category in folders:
    subdir = os.listdir(imgFolder + category)
    print(category)
    
    if not os.path.exists(destFolder + category):
	for image in subdir:
	    img = Image.open(imgFolder + category + '/' + image)
	    img = img.resize((224,224),PIL.Image.ANTIALIAS)
	    if not os.path.exists(destFolder + category):
		os.makedirs(destFolder + category)
	    img.save(destFolder + category + '/' + image)
