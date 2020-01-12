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
import datetime as dt
import argparse
import warnings
import sys
import yaml
import logging

info=None
warning=None
debug=None

def goes_2_roi(loaded_goes, 
               source_area, 
               area_target_def,
               data_key='Rad',
               radius_of_influence=50000):
    """Function that goes from loaded GOES data to data resampled in a projection for an extent"""
    #dat = loaded_goes.metpy.parse_cf(data_key)
    # One long line to avoid memory allocation
    return image.ImageContainerNearest(loaded_goes[data_key].data, 
                source_area, 
                radius_of_influence=radius_of_influence).resample(area_target_def).image_data


def build_geometry(geos_crs,
               source_rows,
               source_cols,
               target_extent,
               target_rows,
               target_cols,
               cartopy_target_proj):
    cartopy_source_extent = geos_crs.x_limits + geos_crs.y_limits
    pyresample_source_extent = (cartopy_source_extent[0],
                                cartopy_source_extent[2],
                                cartopy_source_extent[1],
                                cartopy_source_extent[3])
    source_area = geometry.AreaDefinition('GOES-1X', 'Full Disk','GOES-1X', 
                                          geos_crs.proj4_params,
                                          source_rows, source_cols,
                                          pyresample_source_extent)
    area_target_def = geometry.AreaDefinition('areaTest', 'Target Region', 'areaTest',
                                        cartopy_target_proj.proj4_params,
                                        target_rows, target_cols,
                                        target_extent)
    return source_area, area_target_def
    

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
    fk1 = satim['planck_fk1'].data[0] if len(satim['planck_fk1'].data)> 1 else satim['planck_fk1'].data
    fk2 = satim['planck_fk2'].data[0] if len(satim['planck_fk2'].data)> 1 else satim['planck_fk2'].data
    bc1 = satim['planck_bc1'].data[0] if len(satim['planck_bc1'].data)> 1 else satim['planck_bc1'].data
    bc2 = satim['planck_bc2'].data[0] if len(satim['planck_bc2'].data)> 1 else satim['planck_bc2'].data
    return {
    'fk1':float(fk1),
    'fk2':float(fk2),
    'bc1':float(bc1),                       
    'bc2':float(bc2)}

def Rad2BT(rad, plancks):
    """Radiances to Brightness Temprature (using black body equation)"""
    # unpack
    fk1, fk2, bc1, bc2 = plancks['fk1'], plancks['fk2'], plancks['bc1'], plancks['bc2']
    invRad = rad**(-1)
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

def getMatchesFromFiles(filenames,product=None, band=None, platform=None, year=None, dayofyear=None, hour=None, minute=None, second=None):
    strPatt = (r'(?P<product>(OR_ABI-L1b-Rad[CFM]|OR_ABI-L2-ACM[CFM]))-M6(C?)(?P<band>\d{0,2})_G(?P<platform>\d{2})_s' + 
        r'(?P<year>\d{4})(?P<dayofyear>\d{3})(?P<hour>\d{2})(?P<minute>\d{2})(?P<seconds>\d{2}).*\.nc')
    patt = re.compile(strPatt)
    kwargs = {'product':product,
              'band':band, 
              'platform':platform, 
              'year':year,
              'dayofyear':dayofyear, 
              'hour':hour, 'minute':minute, 'second':second}
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
                      abiproduct='OR_ABI-L1b-RadF',
                      target_proj = ccrs.Mercator(),
                      overwrite=False,
                      short_circuit=False):
    
    # Reference projection for lon-lat
    if abiproduct[-4:-1]=='Rad':
        data_key='Rad'
    elif abiproduct[-4:-1]=='ACM':
        info('Cloud mask request detected. data_key=BCM, band=""')
        data_key = 'BCM'
        bands = [''] # Cloud mask make doesn't have a band
    else:
        raise Exception('Datatype not implmented.')
    debug(str(ROI))
    pc = ccrs.PlateCarree()
    target_extent_mc_cartopy = trasform_cartopy_extent(ROI['extent'], pc, target_proj)
    target_extent_mc_pyresample = cartopy_pyresample_toggle_extent(target_extent_mc_cartopy)
    
    
    roidays = iterProduct(platforms, years, bands, daysofyear)
    # for attributes get list of files
    debug("About to go through list of atribute tuples")
    for attriblist in roidays:
        info(str(attriblist))
        platform, year, band, dayofyear = attriblist
        # build filename
        xarray_name = ('{abiproduct}-M6C{band}_G{platform}_s' + 
        '{year}{dayofyear}_xr.nc').format(abiproduct=abiproduct,
                                    band=str(band).zfill(2),
                                    platform=str(platform).zfill(2),
                                    year=str(year),
                                    dayofyear=str(dayofyear).zfill(2))
        xarray_out_path = Path(outputpath,xarray_name)
        dirlist = os.listdir(indatapath)
        # if not empty
        if not bands[0]:
            filelist = sorted(getMatchesFromFiles(dirlist,product=abiproduct, platform=platform, year=year, dayofyear=dayofyear))
        else:
            filelist = sorted(getMatchesFromFiles(dirlist,product=abiproduct, band=band, platform=platform, year=year, dayofyear=dayofyear))

        debug('Getting filelist.')
        info("There are {} files that match criteria for {}".format(len(filelist), attriblist))
        #print('\n'.join([fname[0] for fname in filelist]))
        if filelist and (overwrite or not xarray_out_path.exists()):
            # Short circuit for debugging
            if short_circuit:
                filelist = filelist[:2]
            # Go through load each file
            # We know the number of files we have right here so lets commit memory
            # Lets get all the paramters and put asside the memory right now
                        # We need to get the cartopy_crs out of the first file
            data_sz = (ROI['east_west_px'],ROI['north_south_px'],len(filelist))
            with xr.open_dataset(Path(indatapath, filelist[0][0])) as xr_dset:
                try:
                    con_dat = xr_dset.metpy.parse_cf(data_key)
                except Exception as a:
                    print('data_kay:', data_key)
                    print('Path(indatapath, filelist[0][0])', Path(indatapath, filelist[0][0]))
                    print('xr_dset.metpy.parse_cf(): ')
                    print(xr_dset.metpy.parse_cf())
                goes_crs = con_dat.metpy.cartopy_crs
                source_cols, source_rows = xr_dset[data_key].data.shape
            debug('goes_crs: {}'.format(goes_crs))
            source_area, area_target_def = build_geometry(goes_crs,
                                                        source_rows,
                                                        source_cols,
                                                        target_extent_mc_pyresample,
                                                        data_sz[1],
                                                        data_sz[0],
                                                        target_proj)
            lons, lats = area_target_def.get_lonlats()
            
            # Need to get type dtype from first file
            time_dtype = np.dtype('<M8[ns]')
            con_dtype  = np.dtype('float32')
            DQF_dtype  = np.dtype('float32')
            lat_lon_dtype = np.dtype('float64')
            plank_dtype = np.dtype('float64')
            attrs ={'platform':platform,
                    'abiproduct':abiproduct}
            if band:
                attrs['band'] = band
            initDataDict = {
                    data_key:(['x','y','time'],np.empty(data_sz, dtype=con_dtype)),
                    'DQF':(['x','y','time'],np.empty(data_sz, dtype=DQF_dtype))}
            if data_key == 'Rad':
                initDataDict.update(
                    {'planck_fk1':(['time'],np.empty(data_sz[-1], dtype=plank_dtype)),
                    'planck_fk2':(['time'],np.empty(data_sz[-1], dtype=plank_dtype)),
                    'planck_bc1':(['time'],np.empty(data_sz[-1], dtype=plank_dtype)),
                    'planck_bc2':(['time'],np.empty(data_sz[-1], dtype=plank_dtype))})

            xfile = xr.Dataset(initDataDict, 
                    coords={'lon': (['x', 'y'], lons),
                            'lat': (['x', 'y'], lats),
                            'time':np.array([None]*data_sz[-1], dtype=time_dtype)},
                    attrs=attrs)
            # Try to guarentee garbage collection.        
            initDataDict=None


            for ind, fname in  enumerate(filelist):
                single_file_start = dt.datetime.now()
                # For memory best not to do this in one shot
                # forloop means we can load just what we need
                # Making sure we don't create any unnecessary data structures
                # Or leave any file pointers open
                with xr.open_dataset(Path(indatapath, fname[0])) as xr_dset:
                    # Do ROI conversions for each loaded file

                    xfile[data_key].data[:,:,ind] = goes_2_roi(xr_dset, 
                                                        source_area, 
                                                        area_target_def,
                                                        data_key=data_key,
                                                        radius_of_influence=50000)
                    debug('ROI min:{} max:{} mean{}'.format(xfile[data_key].data[:,:,ind].min(), 
                                                            xfile[data_key].data[:,:,ind].max(), 
                                                            xfile[data_key].data[:,:,ind].mean()))
                    xfile.DQF.data[:,:,ind] = goes_2_roi(xr_dset, 
                                                        source_area, 
                                                        area_target_def,
                                                        data_key='DQF',
                                                        radius_of_influence=50000)
                    debug('DFQ min:{} max:{} mean{}'.format(xfile.DQF.data[:,:,ind].min(), xfile.DQF.data[:,:,ind].max(), xfile.DQF.data[:,:,ind].mean()))
                    xfile.time.data[ind]=xr_dset.t.data
                    if data_key=='Rad':
                        xfile.planck_fk1.data[ind] = xr_dset.planck_fk1.data
                        xfile.planck_fk2.data[ind] = xr_dset.planck_fk2.data
                        xfile.planck_bc1.data[ind] = xr_dset.planck_bc1.data
                        xfile.planck_bc2.data[ind] = xr_dset.planck_bc2.data
                single_file_end = dt.datetime.now()
                info('Extracted {} in Elapsed seconds: {}'.format(str(xr_dset.t.data),
                                                        (single_file_end - single_file_start).seconds))
            info('Attempting write of {}'.format(xarray_name))
            # save xarray with file name
            try:
                if xarray_out_path.exists(): #May need to remove on overwrite on windows
                    os.remove(xarray_out_path)
                    info('Should have removed old {}'.fomat(xarray_out_path))
                else:
                    info("xarray_out_path.exists() reported false")
                xfile.to_netcdf(xarray_out_path)
                info('Write of {} successful.'.format(str(xarray_name)))
            except Exception as e:
                #Fall back to not lose data
                warning(str(e))
                warning('Try another name')
                anothername = str(xarray_out_path).split('.')[0]+'_'+str(np.random.randint(1111,9999))+'.nc'
                xfile.to_netcdf(anothername)
                info('Write of {} successful.'.format(str(anothername)))
                
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
    parser.add_argument('--product', help='Currently this is a list of goes 16 or 17 eg. --product=OR_ABI-L1b-RadF')
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
    parser.add_argument('--overwrite', help='Should we overwrite any files found.', action='store_true')
    parser.add_argument('--short-circuit', help='DEBUGGING ONLY: cuts off at 2 files to extract', action='store_true')
    args = parser.parse_args()
    
    # Problem with code
    # If I put the defaults in the command line switches
    # the defaults will over-ride config file
    # but if use enters non-default switch
    # that SHOULD over-ride config file
    # so I need to detect default values
    config = {
        'abiproduct': 'OR_ABI-L1b-RadF',
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
    if args.short_circuit:
        config['short_circuit'] = True
    if args.logfile:
        config['logfile'] = args.logfile
    if args.product:
        config['abiproduct'] = args.product

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
    global debug
    config = parseUserArgs()
    started = dt.datetime.now()
    # identifier for the logging
    Rid = np.random.randint(111111111,999999999)
    logging.basicConfig(filename=config['logfile'],
                        format='%(levelname)s %(asctime)s %(message)s '+str(Rid) + ':', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)
    info=logging.info
    warning=logging.warning
    debug=logging.debug
    info(str(config))
    info("Started run: {}".format(started.strftime('%Y/%m/%D %I:%M %p')))
    del config['logfile'] # This would mess with processFilesToROI
    debug(str(config))
    processFilesToROI(**config)
    finished = dt.datetime.now()
    info("Finished run: {}".format(started.strftime('%Y/%m/%D %I:%M %p')))
    info("Elapsed seconds: {}".format((finished-started).seconds))

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

def mat2pandas(matdata):
    mattimes = np.array(matdata['TIME'][0])
    times = [matlab2datetime(time) for time in mattimes]
    temps = matdata['LWIRTemp'][0]
    df = pd.DataFrame({'Time':times, 'LWIRTemp':temps})
    df.set_index('Time',inplace=True)
    return df


if __name__ == '__main__':
    main()





