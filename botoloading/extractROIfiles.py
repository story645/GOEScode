#!/opt/anaconda3/envs/goesenv/bin/python

import numpy as np
from datetime import datetime, timedelta
from pyproj import Proj
import pyproj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyresample import image, geometry
import netCDF4
import os
import os.path as op
import sys
import metpy
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd
import statsmodels.api as sm
import re
import xarray as xr
from itertools import product as iterProduct
from pathlib import Path
import datetime
import argparse
import warnings
import sys
import yaml
import logging

info=None
warning=None

def goes_2_roi(loaded_goes, 
               target_extent,
               target_rows,
               target_cols,
               cartopy_target_proj,
               data_key='Rad',
               radius_of_influence=50000):
    """Function that goes from loaded GOES data to data resampled in a projection for an extent"""
    dat = loaded_goes.metpy.parse_cf('Rad')
    qual = loaded_goes.metpy.parse_cf('DQF')
    #dat = loaded_goes['Rad'].data
    geos_crs = dat.metpy.cartopy_crs
    cartopy_source_extent = geos_crs.x_limits + geos_crs.y_limits
    pyresample_source_extent = (cartopy_source_extent[0],
                                cartopy_source_extent[2],
                                cartopy_source_extent[1],
                                cartopy_source_extent[3])
    rad = dat.data
    source_area = geometry.AreaDefinition('GOES-1X', 'Full Disk','GOES-1X', 
                                          geos_crs.proj4_params,
                                          rad.shape[1], rad.shape[0],
                                          pyresample_source_extent)
    area_target_def = geometry.AreaDefinition('areaTest', 'Target Region', 'areaTest',
                                        cartopy_target_proj.proj4_params,
                                        target_rows, target_cols,
                                        target_extent)
    geos_con_nn = image.ImageContainerNearest(rad, 
                                            source_area, 
                                            radius_of_influence=radius_of_influence)
    geos_dqf_nn = image.ImageContainerNearest(qual.data, 
                                            source_area, 
                                            radius_of_influence=radius_of_influence)
    lons, lats = area_target_def.get_lonlats()

    # Here we are using pyresample for the remapping
    area_proj_con_nn = geos_con_nn.resample(area_target_def)
    area_proj_dqf_nn = geos_dqf_nn.resample(area_target_def)
    info('Extracted {}'.format(str(dat.t.data)))
    return area_proj_con_nn.image_data,area_proj_dqf_nn.image_data, lons, lats
        
def cartopy_pyresample_toggle_extent(input_extent):
    return np.array(input_extent)[np.array([0,2,1,3])]


def trasform_cartopy_extent(source_extent,source_proj, target_proj):
    target_extent = target_proj.transform_points(source_proj, 
                                                 np.array(source_extent[:2]),
                                                 np.array(source_extent[2:])).ravel()
    # target_extent in 3D, must be in 2D
    return cartopy_pyresample_toggle_extent(np.array(target_extent)[np.array([0,1,3,4])])

def normIm(im,gamma=1.0,reverse=False):
    nim = ((im-np.nanmin(im))*(np.nanmax(im)-np.nanmin(im))**(-1))
    if reverse:#want clouds to be white
        nim = (1.0-nim**(gamma))
    return nim

def getPlankConsts(satim):
    """Extract the planck parameters fk1, fk2, bc1 and bc2 for temperature conversion."""
    return {
    'fk1':float(satim['planck_fk1'].data),
    'fk2':float(satim['planck_fk2'].data),
    'bc1':float(satim['planck_bc1'].data),                       
    'bc2':float(satim['planck_bc2'].data)}

def Rad2BT(rad, plancks):
    """Radiances to Brightness Temprature (using black body equation)"""
    # unpack
    fk1, fk2, bc1, bc2 = plancks['fk1'], plancks['fk2'], plancks['bc1'], plancks['bc2']
    invRad = np.array(rad)**(-1)
    arg = (invRad*fk1) + 1.0
    T = (- bc1+(fk2 * (np.log(arg)**(-1))) )*(1/bc2) 
    return T

def convert2intIfPossible(value):
        try:
            if type(1) != type(value):
                value = int(value)
        except:
            pass
        return value

def getMatchesFromFiles(filenames,band=None, platform=None, year=None, dayofyear=None, hour=None, minute=None, second=None):
    strPatt = (r'.*OR_ABI-L1b-Rad(?P<product>[CFM])-M6C(?P<band>\d{2})_G(?P<platform>\d{2})_s' + 
        r'(?P<year>\d{4})(?P<dayofyear>\d{3})(?P<hour>\d{2})(?P<minute>\d{2})(?P<seconds>\d{2}).*\.nc')
    patt = re.compile(strPatt)
    kwargs = {'band':band, 'platform':platform, 'year':year,'dayofyear':dayofyear, 'hour':hour, 'minute':minute, 'second':second}
    pairs = [(filename, re.match(patt,filename)) for filename in filenames if re.match(patt,filename)]
    filtered = pairs
    for key, value in kwargs.items():
        value = convert2intIfPossible(value)
        if value:
            filtered = [(filename,match) for filename, match in filtered 
                        if convert2intIfPossible(match.group(key)) == value]
    return filtered


def processFilesToROI(platforms,
                      years,
                      bands,
                      daysofyear,
                      ROI,
                      indatapath,
                      outputpath,
                      abiproduct='F',
                      target_proj = ccrs.Mercator(),
                      overwrite=False):
    
    # Reference projection for lon-lat
    info(str(ROI))
    pc = ccrs.PlateCarree()
    target_extent_mc_cartopy = trasform_cartopy_extent(ROI['extent'], pc, target_proj)
    target_extent_mc_pyresample = cartopy_pyresample_toggle_extent(target_extent_mc_cartopy)
    
    
    roidays = iterProduct(platforms, years, bands, daysofyear)
    # for attributes get list of files
    for attriblist in roidays:
        info(str(attriblist))
        platform, year, band, dayofyear = attriblist
        # build filename
        xarray_name = ('OR_ABI-L1b-Rad{abiproduct}-M6C{band}_G{platform}_s' + 
        '{year}{dayofyear}_xr.nc').format(abiproduct=abiproduct,
                                    band=str(band).zfill(2),
                                    platform=str(platform).zfill(2),
                                    year=str(year),
                                    dayofyear=str(dayofyear).zfill(2))
        xarray_out_path = Path(outputpath,xarray_name)
        dirlist = os.listdir(indatapath)
        # if not empty
        filelist = sorted(getMatchesFromFiles(dirlist, band=band, platform=platform, year=year, dayofyear=dayofyear))
        info("There are {} files that match criteria for {}".format(len(filelist), attriblist))
        #print('\n'.join([fname[0] for fname in filelist]))
        if filelist and (overwrite or not xarray_out_path.exists()):
            # Short circuit for debugging
            # filelist = filelist[:2]
            # Go through load each file
            openDsets = [xr.open_dataset(Path(indatapath, fname[0])) for fname in  filelist]
            # Do ROI conversions for each loaded file
            roi_rads_array = [goes_2_roi(openDset,
                   target_extent_mc_pyresample,
                   ROI['north_south_px'],
                   ROI['east_west_px'],
                   target_proj) for openDset in openDsets]
            _, _, lat, lon = roi_rads_array[0]
            raddata = np.stack([roi_rad_array[0] for roi_rad_array in roi_rads_array],axis=-1)
            dqfdata = np.stack([roi_rad_array[1] for roi_rad_array in roi_rads_array],axis=-1)
            times = [openDset.t.data for openDset in openDsets]
            planck_fk1 = [openDset.planck_fk1.data for openDset in openDsets]
            planck_fk2 = [openDset.planck_fk2.data for openDset in openDsets]
            planck_bc1 = [openDset.planck_bc1.data for openDset in openDsets]
            planck_bc2 = [openDset.planck_bc2.data for openDset in openDsets]
            # pack into xarray
            # (note ignoring lon-lats)
            xfile = xr.Dataset({'Rad':(['x','y','time'],raddata),
                                'DQF':(['x','y','time'],dqfdata),
                                 'planck_fk1':(['time'],planck_fk1),
                                 'planck_fk2':(['time'],planck_fk2),
                                 'planck_bc1':(['time'],planck_bc1),
                                 'planck_bc2':(['time'],planck_bc2)}, 
                                 coords={'lon': (['x', 'y'], lon),
                                         'lat': (['x', 'y'], lat),
                                         'time':times},
                                  attrs={'platform':platform,
                                         'abiproduct':'OR_ABI-L1b-Rad'+abiproduct,
                                         'band':band})

            xarray_name = ('OR_ABI-L1b-Rad{abiproduct}-M6C{band}_G{platform}_s' + 
                            '{year}{dayofyear}_xr.nc').format(abiproduct=abiproduct,
                                                        band=str(band).zfill(2),
                                                        platform=str(platform).zfill(2),
                                                        year=str(year),
                                                        dayofyear=str(dayofyear).zfill(2))
            info('Attempting write of {}'.format(xarray_name))
            # save xarray with file name
            xfile.to_netcdf(xarray_out_path)
            info('Write of {} successful.'.format(xarray_name))
        elif (len(filelist) == 0):
            warning('No input files found for {}'.format(attriblist))
        elif (xarray_out_path.exists()):
            warning("File exists. No overwite of {}.".format(str(xarray_out_path)))
        #else: # for debugging
        #    print("filelist is empty:" + str(filelist),)
        #    print(' platform: '+str(platform),)
        #    print(' year: '+str(year),)
        #    print(' band: '+str(band),)
        #    print(' dayofyear: '+str(dayofyear),)

def parseUserArgs():
    parser = argparse.ArgumentParser(description='Process GOES files to extract ROI.')
    parser.add_argument('--platform', help='Currently this is a list of goes 16 or 17 eg. --platform=16,17')
    parser.add_argument('--year', help='Data in which year --year=2019')
    parser.add_argument('--dayofyear', help='Julian day of the year eg. --dayofyear=103')
    parser.add_argument('--band', help='Which bands to process eg. --band=8,9')
    parser.add_argument('--output-dir', help='Where to put processed files --ouput-dir=./output')
    parser.add_argument('--data-dir', help='Where to look for data --ouput-dir=./data')
    parser.add_argument('--config', help='Where to find config file.\nConfig can define the ROI')
    parser.add_argument('--extent', help='Definines a lon-lat bounding box. lonmin,lonmax,latmax,latmin')
    parser.add_argument('--pixels-north-south', help='Number of pixels north-south')
    parser.add_argument('--pixels-east-west', help='Number of pixels east-west')
    parser.add_argument('--logfile', help='Location of the logfile')
    parser.add_argument('--overwrite', help='Should we overwrite any files found.')
    args = parser.parse_args()
    
    # Problem with code
    # If I put the defaults in the command line switches
    # the defaults will over-ride config file
    # but if use enters non-default switch
    # that SHOULD over-ride config file
    # so I need to detect default values
    config = {
        'platforms': [16,17],
        'years': [2019,2020],
        'daysofyear': list(range(1,366)),  
        'bands':  list(range(1,17)),
        'ROI': {
            'extent': None,
            'north_south_px': None,
            'east_west_px': None,
        },
        'outputpath': Path('.'),
        'indatapath': Path('.'),
        'overwrite': False,
        'logfile' : 'extractROI.log'
    }
    config_update={}
    if args.config:
        config_file = Path(args.config) 
        if config_file.exists():
            with open(config_file,'r') as fid:
                try:
                    config_update = yaml.safe_load(fid)
                except yaml.YAMLError as exc:
                    warnings.warn(exc)
                    sys.exit(-1)
        else:
            warnings.warn('Data path does not exist')
            sys.exit(-1)  
    config.update(config_update)

    # Command line switches always override config
    if args.platform:
        config['platforms'] = [int(item) for item in args.platform.split(',')]
    if args.year:
        config['years'] = [int(item) for item in args.year.split(',')]
    if args.dayofyear:
        config['daysofyear'] = [int(item) for item in args.dayofyear.split(',')]
    if args.band:
        config['bands'] = [int(item) for item in args.band.split(',')]
    if args.extent:
        config['ROI']['extent'] = [float(item) for item in args.extent.split(',')]
    if args.pixels_north_south:
        config['ROI']['north_south_px'] = [int(item) for item in args.pixels_north_south.split(',')]
    if args.pixels_east_west:
        config['ROI']['east_west_px'] = [int(item) for item in args.pixels_east_west.split(',')]
    if args.output_dir:
        config['outputpath'] = Path(args.output_dir)
    if args.data_dir:
        config['indatapath'] = Path(args.data_dir)
    if args.overwrite:
        config['overwrite'] = True
    if args.logfile:
        config['logfile'] = args.logfile

    # Just in case paths are strings
    config['outputpath'] = Path(config['outputpath'])    
    config['indatapath'] = Path(config['indatapath'])

    # Check Paths exist
    if not config['outputpath'].exists():
        warnings.warn('Output path does not exist')
        sys.exit(-1)
    if not config['indatapath'].exists():
        warnings.warn('Data path does not exist')
        sys.exit(-1)

    return config


def main():
    global info
    global warning
    config = parseUserArgs()
    started = datetime.datetime.now()
    # identifier for the logging
    Rid = np.random.randint(111111111,999999999)
    logging.basicConfig(filename=config['logfile'],
                        format='%(levelname)s %(asctime)s %(message)s '+str(Rid) + ':', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    info=logging.info
    warning=logging.warning

    info(str(config))
    info("Started run: {}".format(started.strftime('%Y/%m/%D %I:%M %p')))
    del config['logfile'] # This would mess with processFilesToROI
    processFilesToROI(**config)
    finished = datetime.datetime.now()
    info("Finished run: {}".format(started.strftime('%Y/%m/%D %I:%M %p')))
    info("Elapsed seconds: {}".format((finished-started).seconds))
if __name__ == '__main__':
    main()





