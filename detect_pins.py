##/usr/bin/env python

from skimage.feature import match_template, canny, peak_local_max
from skimage.filters import gaussian, median, rank
from skimage.morphology import disk
from skimage.io import imread, imsave, imshow
from skimage import color, exposure
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_ellipse 
from skimage import img_as_float

import matplotlib.pyplot as plt
import numpy as np

import optparse

def unsharp(image, unsharp_strength=0.8, blur_size = 8):
	blurred = gaussian(image, blur_size)
	highpass = image - unsharp_strength * blurred
	sharp = image + highpass
	return sharp

def main():
	parser = optparse.OptionParser()
	parser.add_option('-i', '--image', type=str, default='/Users/smartin/lucy/empty/puck4_five_missing.jpg', help='Specify the image')
	options, args = parser.parse_args()
	fig, axes = plt.subplots(3, 4)
	a = axes[0, 0]
	original_image = img_as_float(imread(options.image))
	a.imshow(original_image)
	a.set_title('original image')
	selem = disk(30)
	gray_image = color.rgb2gray(original_image)

	b = axes[0, 1]
	b.imshow(gray_image)
	b.set_title('gray image')
	
	img_rank = rank.equalize(gray_image, selem=selem)
	c = axes[0, 2]
	c.imshow(img_rank)
	c.set_title('rank equalized image')
	
	d = axes[0, 3]
	img_med = median(gray_image, selem=selem)
	d.imshow(img_med)
	d.set_title('median filtered image')
		
	e = axes[1, 0]
	img_sharp = unsharp(gray_image)
	e.imshow(img_sharp)
	e.set_title('unsharp')

	f = axes[1, 1]
	edges = canny(img_sharp, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
	f.imshow(edges)
	f.set_title('edges from unsharp image')

	g = axes[1, 2]
	result = hough_ellipse(edges, accuracy=10)
	result.sort(order='accumulator', reverse=True)
	img_eli = img_gray.copy()
	for best in result[:10]:
		yc, xc, a, b = [int(round(x)) for x in best[1:5]]
		orientation = best[5]
		# Draw the ellipse on the original image
		cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
		img_eli[cy, cx] = 1.
	g.imshow(img_eli)
	g.set_title('detected ellipses')

	plt.show()

if __name__ == '__main__':
	main()