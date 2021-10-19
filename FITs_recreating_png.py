import glob
import os
import warnings
from astropy.io import fits
import FITs_to_PNG_MW
from PIL import Image
import numpy as np

def photon_counts_from_FITS(imgs, scale_factor=1, bands='grz'):
    """
    Convert a FITS data file into a photon counts array. Begins by defining the softening parameters
    for each of the grz bands (necessary?) befroe creating an empty (3, x, x) array. Each band in 
    the image (grz) is run through the flux_to_counts function, outputting the number of photon counts
    for each pixel of the image in each band. This is appended to the empty array and pased through 
    the poisson_noise function to return a rescaled version of the image. Finally a reverse of the first 
    process is conducted to convert the scaled photon counts back into the original flux format of the
    FITs file so it can be processed into a .png later.

    Inputs:
        imgs - (3, x, x) numpy array of floats with flux values for a specific image in each band
        scale_factor - The factor by which the image is to be scaled (default set to 1)
        bands - Defines the bands over which the image is captured (default set to grz)
    
    Outputs:
        imgs - The original FITs input file of fluxes
        img_counts - The photon counts for the original FITs file before rescaling
        img_scaled - The output fluxes, rescaled for the new distance, returned to the original FITs
        formating 
    """
    
    size = imgs[0].shape[1]
    
    soft_params = dict(g=(0, 0.9e-10),
                     r=(1, 1.2e-10),
                     z=(2, 7.4e-10) #We think this is the right plane order...hopefully (Are these necessary anymore?)
                     )
                  
    img_counts = np.zeros((3, size, size), np.float32)
    for im, band in zip(imgs, bands):  #im is an (x, x)
        plane, softening_parameters = soft_params.get(band, (0, 1.))
        counts = flux_to_counts(im, softening_parameters, band) #im is a (3, x, x) array ; softening_parameters is an int
        img_counts[plane, :, :] = counts #counts is (x, x) in shape
        
    
    new_position_counts = poisson_noise(img_counts, scale_factor, size) #scaling by scale_factor (set 1 by default)
    #Should the scale factor be determined by some luminosity deistances or is a set value good enough?
    #Should the equivilent redshift be calculated here to determine if K-corrections are necessary?

    img_scaled = np.zeros((3, size, size), np.float32)
    for count, band in zip(new_position_counts, bands): #count is a (x, x)
        plane, softening_parameters = soft_params.get(band, (0, 1.))
        nMgys = counts_to_flux(count, band)
        img_scaled[plane, :, :] = nMgys #nMgys is (x, x) in shape
       
    clipped_img = np.clip(imgs, 1e-10, None) #Replaces any values in array less than 1e-10 with 1e-10.

    return clipped_img, img_counts, img_scaled

def flux_to_counts(im, softening_parameters, band, exposure_time_seconds = 90. * 3.): #Check exposure
    """
    Converts data in the form of fluxes in nanomaggies into equivilent photon counts. First defines
    the photon energy assuming a central average wavelength for each band. The FITs data is then 
    clipped to remove any negative values (problematic later when adding the noise - root(N) - and 
    poisson distribution - lambda = N). Nanomaggies are converted to fluxes via a conversion factor,
    and this is then multiplied by the pixel exposure time divided by the energy pf the appropriate band
    to obtain the photon count.

    Inputs:
        im - (x, x) array of FITs data in nMgy's 
        softening parameter - Necessary? (Used in making asinh magnitudes)
        band - The relevent band grz we are working in
        exposure_time_seconds - The collecting time for each pixel, set to 90 seconds by defualt.

    Outputs:
        img_photns - (x, x) array of floats representing photon counts within each pixel
    """
    photon_energy = dict(g=(0, 4.12e-19), #475nm
                         r=(1, 3.20e-19), #622nm
                         z=(2, 2.20e-19) #905nm
                         )# Using wavelength for each band better than a general overall wavelength
    
    size = im.shape[1]
    #img_nanomaggies_nonzero = np.clip(im, 1e-10, None) #Array with no values below 1e-9
    img_nanomaggies_nonzero = im
    img_photons = np.zeros((size, size), np.float32)

    energy = photon_energy.get(band, 1)[1] #.get has inputs (key, value) where value is returned if key deos not exist
    #flux = asinh_mag_to_flux(Im, softening_parameters)
    flux = img_nanomaggies_nonzero * 3.631e-6
    img_photons[:, :] = np.multiply(flux, (exposure_time_seconds / energy)) #the flux values reach the upper limits of float manipulation - need to be scaled by some value for operations to be conducted
    
    
    return img_photons

def counts_to_flux(counts, band, exposure_time_seconds = 90. * 3.):
    """
    Converts data from the form of photon counts into nanomaggy fluxes. Starts by defining the 
    energy for photons in each the 3 bands, using a central band wavelength. Then takes the counts 
    and divides through by the exposure time over the photon energy of the band. This igves the flux 
    in Jy which can then be converted to nMgy's via a scale factor 3.631e-6.

    Inputs:
        counts - (x, x) array of photon counts for each pixel (float)
        band - The relevent band grz we are working in
        exposure_time_seconds - The collecting time for each pixel, set to 90 seconds by defualt.

    Outputs:
        img_mgy - (x, x) array of floats representing the nMgy value for each pixel
    """
    photon_energy = dict(g=(0, 4.12e-19), #475nm
                         r=(1, 3.20e-19), #622nm
                         z=(2, 2.20e-19) #905nm
                         )# TODO assume 600nm mean freq. for gri bands, can improve this
    
    size = counts.shape[1]
    img_flux = np.zeros((size, size), np.float32)
    
    energy = photon_energy.get(band, 1)[1]
    img_flux[:, :] = counts / (exposure_time_seconds / energy)
    
    img_mgy = img_flux / 3.631e-6
        
    return img_mgy

def poisson_noise(photon_count, x, size):
    """
    Scales the photon count by 1/d^2 to account for decreased photon numbers at new position
    before adding a poissonly distributed random noise to each channel for each pixel.

    Inputs:
        photon_count - (x, x) array of photon counts for each pixel
        x - The scale factor we want to scale the image by

    Outputs:
        photon_with_poisson - (x, x) array of floats of the original input data, scaled to the new
        distance and with random poisson noise added.
    """
    photon_at_distance_scale_x = photon_count * (1/x)**2
    #photon_at_distance_scale_x = np.where(photon_count>5e10, photon_count * (1/x)**2, photon_count) #Only scales certain pixels with larger counts, imporves speckling
    photon_with_poisson = photon_at_distance_scale_x + np.random.poisson(np.sqrt(np.abs(photon_at_distance_scale_x)))
    #photon_with_poisson += np.random.poisson(5e12, (3, size, size))
    return photon_with_poisson

if __name__ == '__main__':
    
    dir_name='J000fitstest' #Sets the name of the file the input png's are stored in

    print('\n Begin \n') #Print statemnt to track progress when running code
    imgs = {} #Opens dictionary for storing images
    for filename in glob.iglob(os.getcwd() + '/' + f'{dir_name}' + '/**/*.fits', recursive=True): #operates over all FIT's within the desired directory
        try:
            img, hdr = fits.getdata(filename, 0, header=True) #Extract FITs data
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))
        imgs[filename]=img #Unsure if didctionary would be better here if required for a large quantity of data
        #imgs.append(img)
  


    final_data = {} #Create dictionary to append final data to

    for key in imgs.keys():
        final_data[key.replace('.fits', '').replace(os.getcwd() + '/' + f'{dir_name}' + '/', '')]=photon_counts_from_FITS(imgs[key], scale_factor=2) #Second input is scale factor

    for entry_name in final_data.keys():
        FITs_to_PNG_MW.make_png_from_corrected_fits(final_data[entry_name][0], 'Original_FITS_data_image_' + entry_name + '.png', 424)
        FITs_to_PNG_MW.make_png_from_corrected_fits(final_data[entry_name][2], 'Scaled_FITS_data_image_' + entry_name + '.png', 424)

    print('\n End \n')
