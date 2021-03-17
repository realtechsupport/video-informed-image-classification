#!/usr/bin/env python3
# utlities.py (python3)
# additional helper functions
# Catch+ Release / Return to Bali
# Flask interface for linux computers
# experiments in knowledge documentation; with an application to AI for ethnobotany
# jan 2020
# tested on ubuntu 18 LTS, kernel 5.3.0
#-------------------------------------------------------------------------------
import sys, os, io, time, datetime, glob
import argparse, zipfile
from os import listdir
from os.path import isfile, join
from os import environ, path
import json as simplejson
import subprocess
import psutil, shutil
from shutil import copyfile
import signal, wave
import contextlib
import numpy as np
import math
from random import *
from datetime import datetime
import pytz
from pathlib import Path

#-------------------------------------------------------------------------------
def create_timestamp(item, location):
    tz = pytz.timezone(location)
    now = datetime.now(tz)
    current_time = now.strftime("%d-%m-%Y-%H-%M")
    stamp_item =  current_time + '_' + item
    return(stamp_item)

#------------------------------------------------------------------------------
def zipit(path, zfile):
    if(len(os.listdir(path)) < 1):
        print('nothing to archive...')
    else:
        with zipfile.ZipFile((path + zfile), 'w') as zipObj:
            print('archiving....')
            for foldername, subfolders, filenames in os.walk(path):
                for filename in filenames:
                    #create complete filepath of file in directory
                    filePath = os.path.join(foldername, filename)
                    zipObj.write(filePath)

#-------------------------------------------------------------------------------
def removefiles(app, patterns, locations, exception):
    for loc in locations:
        os.chdir(app.config[loc])
        for pat in patterns:
            for file in glob.glob(pat):
                if(exception in file):
                    pass
                else:
                    os.remove(file)

    shutil.rmtree(app.config['IMAGES'])

#------------------------------------------------------------------------------
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
#-------------------------------------------------------------------------------
def write2file(filename, comment):
    file = open(filename, "a")
    value = file.write(comment + '\n')
    file.close()
#-------------------------------------------------------------------------------
def write2file_m(filename, comment, method):
    file = open(filename, method)
    value = file.write(comment)
    file.close()
#-------------------------------------------------------------------------------
def rename_all(source, offset):
    #weird condition - adding ext to name in one step results in numbering errors
    #cause not known to me.. two steps workaround
    source = source + '/'
    ext = ".jpg"

    for count, filename in enumerate(os.listdir(source)):
        n = offset + count
        name = str(n)
        name_s = f'{name:0>4}'
        tsrc = source + filename
        dst = source + name_s
        os.rename(tsrc, dst)

    for count, filename in enumerate(os.listdir(source)):
        ext_filename = filename + ext
        tsrc = source + filename
        dst = source + ext_filename
        os.rename(tsrc, dst)

#-------------------------------------------------------------------------------
def downloadweather_check(currenturl, referenceurl, referencefile, target, location, targetfile):
    try:
        file = wget.download(currenturl, target)
        if os.path.exists(target):
                shutil.move(file, target)
        #get the reference data only once
        path, dirs, files = next(os.walk(location))
        if(referencefile in files):
            print('\nalready downloaded the data..\n')
            pass
        else:
            wget.download(referenceurl, targetfile)

    except:
        'data download failed...'
        pass

#-------------------------------------------------------------------------------
def downloadassets_check(current_url, location, filename):

    targetfile = os.path.join(location, filename)
    asset_saved = False
    try:
        path, dirs, files = next(os.walk(location))
        if(filename in files):
            print('\nalready downloaded the data..\n')
            asset_saved = True
            pass
        else:
            wget.download(current_url, targetfile)
            asset_saved = True

    except:
        asset_saved = False
        pass

    return(asset_saved)
#-------------------------------------------------------------------------------
def index2name(indeces, names):
    collection = []

    if(isinstance(indeces, int)):
        found = names[indeces]
        collection.append(found)
    else:
        for index in range(len(indeces)):
        #for index in range(len(indeces) - 1):
            #print('the info is: ', indeces, index)
            found = names[indeces[index]]
            collection.append(found)

    return(collection)
#---------------------------------------------------------------------------------
def unique(data):
    ulist = []
    for x in data:
        if (x not in ulist):
            ulist.append(x)

    return(ulist)
#---------------------------------------------------------------------------------
