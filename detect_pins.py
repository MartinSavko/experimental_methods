#!/home/experiences/proxima2a/com-proxima2a/Enthought/Canopy_64bit/User/bin/python

from skimage.feature import match_template, canny, peak_local_max
from skimage.filters import gaussian, median, rank, threshold_otsu
from skimage.morphology import disk
from skimage.io import imread, imsave, imshow
from skimage import color, exposure
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_ellipse 
from skimage import img_as_float, img_as_int

import matplotlib.pyplot as plt
import numpy as np

import optparse
import glob
import random

def unsharp(image, unsharp_strength=0.8, blur_size = 8):
    blurred = gaussian(image, blur_size)
    highpass = image - unsharp_strength * blurred
    sharp = image + highpass
    return sharp

def combine(template='lid3_empty_*jpg', start=0, length=30):
    i = np.zeros((576, 768, 3), dtype=np.uint16)
    images = glob.glob(template)
    imgs = [imread(img) for img in images[start:start+length]]
    for img in imgs:
        i += img
    return i
    
    
def test():
    
    image_series = glob.glob('full_dewar/puck*_*in*_200.jpg')
    templates = [n.replace('200', '*') for n in image_series]
    template_empty = imread('template_empty.jpg')
    h, w = template_empty.shape
    
    print 'len(templates)', len(templates)
    fig, axes = plt.subplots(3, 4)
    a = axes.ravel()
    k = 0
    used = []
    while k<12:
    #for template in templates[:12]:
        template = random.choice(templates)
        if template in used:
            pass
        else:
            used.append(template)
            original_image = img_as_float(combine(template, length=200))
            ax = a[k]
            gray_image = color.rgb2gray(original_image)
            img_sharp = unsharp(gray_image)
            edges = canny(img_sharp, sigma=3.0, low_threshold=0.04, high_threshold=0.05)
            med_unsharp = median(img_sharp/img_sharp.max(), selem=disk(4))
            sharp_med_unsharp = unsharp(med_unsharp)
            edges_med = canny(sharp_med_unsharp, sigma=7)
            match = match_template(gaussian(edges_med, 4), template_empty)
            print 'match.max()'
            print match.max()
            peaks = peak_local_max(gaussian(match, 3), threshold_abs=0.3, indices=True)
            print 'template', template
            print '# peaks', len(peaks)
            print peaks
            ax.imshow(original_image) #, cmap='gray')
            #ax.imshow(gaussian(edges_med, 3), cmap='gnuplot')
            for peak in peaks:
                y, x = peak
                rect = plt.Rectangle((x, y), w, h, edgecolor='g', linewidth=2, facecolor='none')
                ax.add_patch(rect)
            #ax[edges] = (0, 1, 0)
            #image = img_as_int(original_image)
            #image[edges==True] = (0, 255, 0)
            ax.set_title(template.replace('full_dewar/', '').replace('_*.jpg', '') + ' detected %s' % (16-len(peaks),))
            k += 1
    plt.show()
    
def main():
    parser = optparse.OptionParser()
    parser.add_option('-i', '--image', type=str, default='empty_dewar/puck4_five_missing.jpg', help='Specify the image')
    parser.add_option('-c', '--combine', type=int, default=1, help='Specify the number of images to combine')
    options, args = parser.parse_args()
    
    if options.combine == 1:
        original_image = img_as_float(imread(options.image))
    else:
        original_image = img_as_float(combine(options.image, length=options.combine))
    
    fig, axes = plt.subplots(3, 4)
    a = axes[0, 0]
    a.imshow(original_image)
    a.set_title('original image')
    selem = disk(30)
   
    gray_image = color.rgb2gray(original_image)
    b = axes[0, 1]
    b.imshow(gray_image, cmap='gray')
    b.set_title('gray image')

    img_rank = rank.equalize(gray_image, selem=selem)
    c = axes[0, 2]
    c.imshow(img_rank, cmap='gray')
    c.set_title('rank equalized image')

    edges = canny(img_rank, sigma=5)
    #img_med = median(gray_image, selem=selem)
    d = axes[0, 3]
    d.imshow(edges, cmap='gray')
    d.set_title('edges from rank equalized')
        
    e = axes[1, 0]
    img_sharp = unsharp(gray_image)
    img_sharp = img_sharp/float(img_sharp.max())
    e.imshow(img_sharp, cmap='gray')
    imsave('img_sharp.jpg', img_sharp)
    e.set_title('unsharp')
    
    f = axes[1, 1]
    edges = canny(img_sharp, sigma=7.0, low_threshold=0.04, high_threshold=0.05)
    f.imshow(gaussian(edges, 3), cmap='gray')
    f.set_title('edges from unsharp image sigma=7')
     
    g = axes[1, 2]
    edges = canny(img_sharp, sigma=2.0, low_threshold=0.04, high_threshold=0.05)
    g.imshow(gaussian(edges, 3), cmap='gray')
    g.set_title('edges from unsharp image sigma=2')
    
    h = axes[1, 3]
    edges = canny(img_sharp, sigma=3.0, low_threshold=0.04, high_threshold=0.05)
    h.imshow(gaussian(edges, 3), cmap='gray')
    h.set_title('edges from unsharp image sigma=3')
    
    i = axes[2, 0]
    edges = canny(img_sharp, sigma=4.0, low_threshold=0.04, high_threshold=0.05)
    i.imshow(gaussian(edges, 3), cmap='gray')
    i.set_title('edges from unsharp image sigma=4')
    #j = axes[2, 1]
    #j.imshow(gaussian(img_sharp, sigma=4), cmap='gray')
    #j.set_title('gaussian on unsharp sigma=4')
    
    imsave('edges.jpg', img_as_int(edges))
    print edges
    #j = axes[2, 1]
    #result = hough_ellipse(edges, min_size=20, max_size=100)
    #result.sort(order='accumulator', reverse=True)
    #print 'result', result
    #img_eli = img_gray.copy()
    #for best in result[:1]:
        #yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        #orientation = best[5]
        ## Draw the ellipse on the original image
        #cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        #img_eli[cy, cx] = 1.
    #g.imshow(img_eli)
    #g.set_title('detected ellipses')
    
    k = axes[2, 2]
    med_unsharp = median(img_sharp/img_sharp.max(), selem=disk(10))
    k.imshow(med_unsharp, cmap='gray')
    k.set_title('median on unsharp')
    
    sharp_med_unsharp = unsharp(med_unsharp)
    l = axes[2, 3]
    edges_med = canny(sharp_med_unsharp, sigma=7) #, high_threshold=0.2)
    #edges_med = gaussian(edges_med, 7)
    imsave('edges_med.jpg', img_as_int(edges_med))
    l.imshow(gaussian(edges_med, 3), cmap='gray')
    l.set_title('edges from med unsharp')
    

 
    #abcdefghijkl
    ##i = axes[2,0]
    ##i.imshow(original_image[:,:,0], cmap='gray')
    ##i.set_title('red channel')
    
    ##j = axes[2,1]
    ##j.imshow(original_image[:,:,1], cmap='gray')
    ##j.set_title('green channel')
    
    ##k = axes[2,2]
    ##k.imshow(original_image[:,:,2], cmap='gray')
    ##k.set_title('blue channel')
    
    plt.show()

if __name__ == '__main__':
    #main()
    test()