import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from astropy import convolution as conv
<<<<<<< HEAD
from astropy import cosmology as cosmo
=======
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
>>>>>>> main
from scipy import signal as sig
import random

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

# read in galaxy:
with open('inputgalaxies.npy','rb') as f:
    gal_input = np.load(f)
    input_redshift = np.load(f)
        
with open('targetgalaxies.npy','rb') as g:
    gal_target = np.load(g)
    output_redshift = np.load(g)

n = gal_input.shape[0]
m = random.randint(0,n-1)

image = gal_input[m,...] # testing with one input image (all filters)

sd = 1.5 # std dev of gaussian
seeing = 2.354*sd # FWHM ~ 2.354*sd
<<<<<<< HEAD

# observing simulated galaxy:
# resize object to fit kernel
def observe_gal(image, input_redshift, output_redshift, seeing):
    image = resize(image, input_redshift, output_redshift)
    return image


=======

# observing simulated galaxy:
# resize object to fit kernel
# def observe_gal(image, input_redshift, output_redshift, seeing):
#     image = resize(image, input_redshift, output_redshift)
#     return image


>>>>>>> main
# convolving image with gaussian psf:
def convolve_psf(image, seeing):
    psf = conv.Gaussian2DKernel(seeing, x_size=60, y_size=60)
    plt.imshow(psf.array, interpolation='none', origin='lower')
    convolved = np.empty((17, 60, 60))
<<<<<<< HEAD
=======
    # convolve PSF with image in all 17 filters
>>>>>>> main
    for i in range(17):
        convolved[i] = sig.convolve2d(image[...,i], psf.array, mode = 'same')
    return convolved

convolved_image = convolve_psf(image,seeing)

<<<<<<< HEAD
=======
# plot image before PSF in all filteres
>>>>>>> main
for j in range(17):
    plt.subplot(3,6,j+1)
    plt.imshow(image[...,j], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' before PSF') 
plt.show  

<<<<<<< HEAD
=======
# plot image after PSF in all filters
>>>>>>> main
for k in range(17):
    plt.subplot(3,6,k+1)
    plt.imshow(convolved_image[k], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' after PSF') 
plt.show()   


# changes to brightness:
def dimming(image, input_redshift, output_redshift):
    d_i = cosmo.luminosity_distance(input_redshift)
    d_o = cosmo.luminosity_distance(output_redshift)
    dimming = (d_i / d_o)
    dimmed = convolved_image*dimming
    return dimmed

dimmed = dimming(convolved_image, input_redshift, output_redshift)

# plot dimmed image in all filters
for l in range(17):
    plt.subplot(3,6,l+1)
    plt.imshow(dimmed[l], interpolation='none', origin='lower', cmap='inferno', vmin=0, vmax=1)
    plt.axis('off') 
plt.suptitle('Image ' + str(m) + ' at redshift = ' + str(input_redshift) + ' after PSF and dimming') 
plt.show()  

# changes to size:
def rebinning(image, input_redshift, output_redshift):
    d_i = cosmo.luminosity_distance(input_redshift)
    d_o = cosmo.luminsoity_distance(output_redshift)
    scale_factor = (d_i / (1 + input_redshift)**2) / (d_o / (1 + output_redshift)**2)
    rebinned = zoom(image, scale_factor)
    return rebinned
  
  # does this need to be normalised by dividing by the sum of the image/flux?
    
# adding shot noise (from variations in the detection of photons from the source):
def add_shot_noise(image, output_exptime):            
    # shot_noise = np.sqrt(convolved * output_exptime) * np.random.poisson()
    with_shot_noise = np.random.poisson(image)
    return with_shot_noise

# adding background noise (from numerous sources - i.e. the sky, electrons in detector which are appearing randomly from thermal noise): 
def add_background(image):
    # code to generate background noise, background = ...
<<<<<<< HEAD
    with_background = np.random.normal(mean=0, peak, image)
=======
    with_background = np.random.normal(...)
>>>>>>> main
    return with_background

# saving data for use in VAE:
def save_data(image):
    # code to save data for VAE - need to edit
    return image