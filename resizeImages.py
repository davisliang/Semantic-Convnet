import PIL
from PIL import Image
import os
width = 224

folders = os.listdir('caltech95Val')
for category in folders:
    subdir = os.listdir('caltech95Val/' + category)
    print(category)
    for image in subdir:
        img = Image.open('caltech95Val/' + category + '/' + image)
	img = img.resize((224,224),PIL.Image.ANTIALIAS)
	if not os.path.exists('resizedCaltechVal/' + category):
	    os.makedirs('resizedCaltechVal/' + category)
	img.save('resizedCaltechVal/' + category + '/' + image)
