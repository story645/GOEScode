#!/opt/anaconda3/envs/goesenv/bin/python

from extractROIfiles import processFilesToROI
import datetime

def main():
    # Get data directory
    indatapath= '/mnt/g/GOESProj/data/fall_data/'
    #indatapath = '/mnt/g/GOESProj/data/sharedData/scratch/april_data'
    outputPath='/mnt/g/GOESProj/data/roi_fall_data/'
    # Get platforms to process
    platforms = [16, 17]
    platforms = [16]
    # Get years to process
    years = [2019]
    # Get bands to process
    bands = [ 8, 9, 10]
    #bands = [7,8]
    bands = [8,9,10]
    # Get days to process
    daysofyear = [243, 244, 245]
    #daysofyear = [103,102,101]
    #daysofyear = [103]
    # Target ROI
    ROI ={
        'extent':[-109.59326, -102.40674, 8.94659, -8.94656],
        'north_south_px': 2001,
        'east_west_px':401
    }
    started = datetime.datetime.now()
    print("started ", started.strftime('%Y/%m/%D %I:%M %p'))
    processFilesToROI(platforms,
                        years,
                        bands,
                        daysofyear,
                        ROI,
                        indatapath,
                        outputPath)
    finished = datetime.datetime.now()
    print('Finished ', started.strftime('%Y/%m/%D %I:%M %p'))
    print('Elapsed seconds', (finished-started).seconds)

if __name__ == '__main__':
    main()