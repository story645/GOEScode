import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from cartopy import crs as ccrs
from goesutils import goes_2_roi, radiance_to_BT, cartopy_pyresample_toggle_extent, trasform_cartopy_extent
from goesutils import norm_im
import metpy
from scipy import stats

def make_roi_from_nc (inputpath,outputpath, 
                      sat,
                      day,
                      band,
                      mc_rows,
                      mc_cols,
                      target_extent_mc_pyresample,
                      rad_prefix='OR_ABI-L1b-RadF-M6C',
                      overwrite=False):
    mc = ccrs.Mercator()
    if band:
        filetemplate = f'{rad_prefix}{band:02}_G{sat:02}_s2019{day}*.nc'
        datakey='Rad'
    else:
        filetemplate = f'{rad_prefix}_G{sat:02}_s2019{day}*.nc'
        datakey='BCM'
    allfiles = list(inputpath.glob(filetemplate))
    for infilename in allfiles:
        infilename_no_nc = str(infilename).split('.')[0]
        outfile = Path(outputpath)/Path(infilename_no_nc).parts[-1]
        if overwrite or (not Path(str(outfile)+'.npy').exists()):

            dataset = xr.open_dataset(infilename)
            roi_rads = goes_2_roi(dataset,
                           target_extent_mc_pyresample,
                           mc_rows,
                           mc_cols,
                           mc,
                           datakey)
            np.save(outfile,roi_rads)
            dataset.close()
            print(f'Writing file: {str(outfile)+".npy"}')
        else:
            print('File exists')

def main():
    day = 103
    band = 8
    sat = 16
    npPath = Path('/tmp/npOut/all_npy3')
    ncPath = Path('/sharedData/scratch/')
    mc_rows = 1001
    mc_cols = 401
    #mc_rows = 500
    #mc_cols = 200
    #Projections
    pc = ccrs.PlateCarree()
    mc = ccrs.Mercator()
    #ROI
    extent_pc = [-109.59326, -102.40674, 8.94659, -8.94656]

    # Set up mc extents
    target_extent_mc_cartopy = trasform_cartopy_extent(extent_pc, pc, mc)
    target_extent_mc_pyresample = cartopy_pyresample_toggle_extent(target_extent_mc_cartopy)
    make_roi_from_nc(Path(ncPath),
                     npPath,
                     sat,
                     day,
                     band,
                     mc_rows,
                     mc_cols,
                     target_extent_mc_pyresample)

if __name__ == '__main__':
    main()
