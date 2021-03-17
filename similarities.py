#!/usr/bin/env python3
# similarities.py (python3)
# Catch+ Release / Return to Bali
# routines to check image similarities / qualities
# with 2 methods, MSE and SSIM
# https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
#-------------------------------------------------------------------------------

from utilities_v2 import *
from skimage import measure
#from skiimage 0.18 on...compare_ssim is in metrics.structual_similarity
from skimage import metrics
from PIL import Image, ImageStat
#-------------------------------------------------------------------------------

def get_mse(imagename1, imagename2):
	# the 'Mean Squared Error' between  two images: sum of the squared differences
	# the two images must have the same dimension
	# larger values = LESS similar; 0 = identical images

	try:
		img1 = Image.open(imagename1).convert('RGB')
		img1 = np.array(img1)
		img2 = Image.open(imagename2).convert('RGB')
		img2 = np.array(img2)

		err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
		err /= float(img1.shape[0] * img2.shape[1])
		err = float("{0:.2f}".format(err))
		# return the MSE, the lower the error, the more "similar  the two images are
	except:
		err = -1

	return (err)

#------------------------------------------------------------------------------
def get_ssim(imagename1, imagename2):
	#structural similarity Index (ssim)
	#make images compatible to the ssim function: CV2 format
	#returns values between -1 and 1
	#larger values  = MORE similar; 1 = identical images
	try:
		img1 = Image.open(imagename1).convert('RGB')
		img2 = Image.open(imagename2).convert('RGB')
		img1 = np.array(img1)
		img2 = np.array(img2)
		img1 = img1[:, :, ::-1].copy()
		img2 = img2[:, :, ::-1].copy()
		# compute structural similarity
		# returns values between 0 and 1; smaller values = more similar
		s = measure.compare_ssim(img1, img2, multichannel=True)
		#s = metrics.structual_similarity.compare_ssim(img1, img2, multichannel=True)
		s = float("{0:.6f}".format(s))
	except:
		s = -1

	return(s)

#------------------------------------------------------------------------------
def compare2images(imagename1, imagename2):
	m = get_mse(imagename1, imagename2)
	s = get_ssim(imagename1, imagename2)
	return(m, s)

#-------------------------------------------------------------------------------
def brightness(image_file):
	im = Image.open(image_file).convert('L')
	stat = ImageStat.Stat(im)
	return (stat.mean[0])

#-------------------------------------------------------------------------------
def perceived_brightness(image_file):
	im = Image.open(image_file)
	stat = ImageStat.Stat(im)
	gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
		for r,g,b in im.getdata())
	return (sum(gs)/stat.count[0])

#-------------------------------------------------------------------------------
def remove_fuzzy_over_under_exposed(reference, img_loc, collection, ssim_min, lum_max, lum_min):
	#select a 'crisp' image as reference; fuzzy is structural deviation from that image...
	ssim_max = 1.00;
	bad = 0;

	r_brightness = brightness(img_loc + reference)
	max_brightness = (lum_max/100)*r_brightness
	min_brightness = (lum_min/100)*r_brightness
	ssim_min = ssim_min/100;

	for k in range(0, len(collection)):
		temp = img_loc + collection[k]
		t_brightness = brightness(temp)
		result = compare2images((img_loc + reference), temp)
		#print(collection[k], result[1], t_brightness, r_brightness)
		if(result[1] == -1):
			pass
		else:
			if((result[1] > ssim_min) and (result[1] < ssim_max)):
				print("deleting poor image based on ssim results: ", collection[k], result[1])
				os.remove(temp)
				bad = bad+1
			if((t_brightness > max_brightness) or (t_brightness < min_brightness)):
				print("deleting poor image based on brightness: ", collection[k], t_brightness)
				os.remove(temp)
				bad = bad+1

	return(bad)
#-------------------------------------------------------------------------------
