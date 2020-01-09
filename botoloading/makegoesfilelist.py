#!/usr/local/anaconda3/bin/python
""" This script is designed to get a list of all the filenames of GOES files on AWS satisfying 
some bounds on time, bands, and platform
"""

import xarray as xr
import requests
import netCDF4
import boto3
from itertools import product
from subprocess import Popen, PIPE
from botocore import UNSIGNED
from botocore.client import Config
import time

def get_s3_keys(bucket, prefix = ''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    kwargs = {'Bucket': bucket}

    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix):
                yield key

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def make_prefix(product_name, year, day_of_year, hour, band):
    return (product_name+'/'+ str(year) + '/' + str(day_of_year).zfill(3) 
                                         + '/' + str(hour).zfill(2) + '/OR_'+ 
                                         product_name + '-M6C' + str(band).zfill(2))

def get_file_names(bucket_names, product_names, years, days_of_year, hours=None, bands=None, mode=6):
    if hours == None:
        hours = range(24)
    if bands == None:
        bands = range(1,17)
    values = product(bucket_names, product_names, years, days_of_year,hours,bands)
    keys = ['bucket_name', 'product_name', 'year', 'day_of_year', 'hour', 'band', 'mode']
    rows = [ dict(zip(keys,value)) for value in values]
    fnames = []
    for row in rows:
        prefix = make_prefix(row['product_name'], row['year'], row['day_of_year'], row['hour'], row['band'])
        keys = get_s3_keys(row['bucket_name'], prefix)
        fnames.append((row['bucket_name'],keys))
    # Returns a list of generators
    return fnames


def main():
    bucket_names = ['noaa-goes16','noaa-goes17']
    product_names = ['ABI-L1b-RadF']
    years = [2019] 
    days_of_year = range(240,250) # From Tim, these are the fall files
    hours = None
    bands = range(7,11)
    goes_list_of_objects = 'rclone_granules.txt'
    objects = get_file_names(bucket_names, product_names, years, days_of_year, hours, bands)
    if len(objects):
        with open(goes_list_of_objects,'w+') as fid:
            bucket_name, keys = objects[0]
            fid.write('\n'.join([bucket_name + '/' + key for key in keys])+'\n')
    if len(objects)>1:
        for bucket_name, keys in objects[1:]:
            time.sleep(2)
            with open(goes_list_of_objects,'a+') as fid:
                fid.write('\n'.join([bucket_name + '/' + key for key in keys])+'\n')    
    
if __name__ == '__main__':
    main()