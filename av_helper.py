#!/usr/bin/env python3
# av_helper.py
# utilities for audio and video processing
# Catch+ Release / Return to Bali
# Flask interface for linux computers
# experiments in knowledge documentation; with an application to AI for ethnobotany
# jan 2020
# tested on ubuntu 18 LTS, kernel 5.3.0
#-------------------------------------------------------------------------------
import sys, os, io, time, datetime, re
from os import environ, path
import subprocess
import signal, wave
import contextlib
from time import strftime, gmtime
from datetime import datetime, timedelta

#-------------------------------------------------------------------------------
def remove_audio_from_video(videofile):
    videoname = videofile.split('.')
    videofile_new = videoname[0] + '_noaudio.' + videoname[1]
    command = 'ffmpeg -loglevel panic -i ' + videofile + ' -y -vcodec copy -an ' + videofile_new
    subprocess.call(command, shell=True)
    print('new video without audio is now in the same directory as the original')
#------------------------------------------------------------------------------
def extract_audio_from_video(videofile, encoding):
    #encoding: wav, mp3
    videoname = videofile.split('.')
    audiofile = videoname[0] + '.' + encoding
    command = 'ffmpeg -y -i ' + videofile  + ' -f ' + encoding + ' -ab 192000 -vn ' +  audiofile
    subprocess.call(command, shell=True)
    print('\n\naudio track is now in the same directory as the video')
    return(audiofile)
#-------------------------------------------------------------------------------
def get_video_length(filename):
    result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    info  =  [x for x in result.stdout.readlines() if b'Duration' in x]
    duration = info[0].decode('utf-8')
    duration = duration.split(', ')
    time_s = duration[0].split('Duration: ')
    time_t = str(time_s[1])
    return (time_t)
#------------------------------------------------------------------------------
def get_video_resolution(filename):
    result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    info  =  [x for x in result.stdout.readlines() if b'Video' in x]
    resolution = info[0].decode('utf-8')
    resolution = re.findall("(\d+x\d+)", resolution)
    resolution = resolution[-1]
    return (resolution)
#------------------------------------------------------------------------------
def hms_to_seconds(t):
    h, m, s = [float(i) for i in t.split(':')]
    return 3600*h + 60*m + s
#-------------------------------------------------------------------------------
def seconds_to_hms(t):
    #updated to get 2 decimal spaces...
    t1000 = t*1000
    s, ms = divmod(t1000, 1000)
    hms = '%s.%02d' % (time.strftime('%H:%M:%S', time.gmtime(s)), ms)
    return (hms)
#-------------------------------------------------------------------------------
def mic_info(mic):
    if(mic == 'default'):
        cardn = 0
        devicen = 0
    else:
        try:
            command = 'arecord -l | grep ' + mic
            result = (subprocess.check_output(command, shell=True)).decode('utf-8')
            cpos = result.find('card'); dpos = result.find('device')
            devicen = result[(len('device')) + dpos + 1]; cardn = result[(len('card')) + cpos + 1]

        except:
            cardn = 0
            devicen = 0

    return(cardn, devicen)
#-------------------------------------------------------------------------------
def get_fps(videofile):
    command = 'ffprobe -v 0 -of csv=p=0 -select_streams 0 -show_entries stream=r_frame_rate ' + videofile
    result = (subprocess.check_output(command, shell=True)).decode('utf-8')
    r = result.split('/')
    result = round(int(r[0]) / int(r[1]))
    return (result)
#-------------------------------------------------------------------------------
def slice_video(datapath, videofile, duration, destination):
    #sloppy slice... not reencoding...
    os.chdir(datapath)
    vname = videofile.split('.')
    name = vname[0]
    name = destination + name
    command = 'ffmpeg -loglevel panic -i ' + videofile + ' -c copy -map 0 -segment_time ' + str(duration) + ' -f segment -reset_timestamps 1 ' + name + '.'+ '%02d.mp4'
    subprocess.call(command, shell=True)
#-------------------------------------------------------------------------------
def slice_audio(datapath, audiofile, duration, destination):
    #sloppy slice... not reencoding...
    os.chdir(datapath)
    vname = audiofile.split('.')
    name = vname[0]
    name = destination + name
    command = 'ffmpeg -loglevel panic -i ' + audiofile + ' -c copy -map 0 -segment_time ' + str(duration) + ' -f segment -reset_timestamps 1 ' + name + '.'+ '%02d.wav'
    subprocess.call(command, shell=True)
#-------------------------------------------------------------------------------
def make_slices_from_audio(datapath, videofile, audioinput, startsecs, duration, chunk, destination, onechan):
    starts = []
    google_limit = 59
    os.chdir(datapath)
    vname = audioinput.split('.')
    name = vname[0]
    name = destination + name + '_audio'

    len = get_video_length(videofile)
    len_s = hms_to_seconds(len)
    if (chunk >= google_limit):
        chunk = google_limit

    first = seconds_to_hms(startsecs)
    starts.append(first)
    s = startsecs
    for i in range(1, (duration- chunk), chunk):
        s = s + chunk
        start = seconds_to_hms(s)
        starts.append(start)

    #print('here are the start times: ', starts)

    i = 0
    for segment in starts:
        print('Slicing the original audio into chunks ...')
        command = 'ffmpeg -loglevel quiet -y -i ' + audioinput + ' -ss ' + str(segment) + ' -t ' + str(chunk) + ' -acodec copy ' + name + '_' + str(i) + '.wav'
        #print(command)
        subprocess.call(command, shell=True)
        if(onechan == True):
            command = "ffmpeg -loglevel quiet -y -i " + name + '_' + str(i) + '.wav'  + " -loglevel quiet -y -ab 160k -ac 1 -ar 16000 -vn " + name + '_1ch_16k_' + str(i) +'.wav'
            subprocess.call(command, shell=True)
            #wait a moment
            time.sleep(0.25)
            #remove original stereo files..
            os.remove( name + '_' + str(i) + '.wav')

        i = i+1

    return(starts)
#-------------------------------------------------------------------------------
def extract_section(datapath, avfilename, start, end, outputname):
    #start and end in this format: '00:00:00'; save to same location as input
    os.chdir(datapath)
    command = 'ffmpeg -loglevel panic -i -y ' + avfilename + ' -ss  ' + str(start) + ' -to ' + str(end)  + ' -c copy ' + outputname
    subprocess.call(command, shell=True)
#-------------------------------------------------------------------------------
def extract_image(datapath, videofile, moment, quality, imagename):
    os.chdir(datapath)
    command = 'ffmpeg -loglevel panic -ss ' + moment + ' -i ' + videofile + ' -vframes 1 -q:v 1 ' + imagename
    subprocess.call(command, shell=True)
#-------------------------------------------------------------------------------
def get_audio_info(audiofile):
    with contextlib.closing(wave.open(audiofile,'r')) as f:
        rate = f.getframerate()
        frames = f.getnframes()
        duration = frames / float(rate)
    return( rate, frames, duration)
#-------------------------------------------------------------------------------
def convert_mp4_to_webm(video_mp4):
    videoname = video_mp4.split('.mp4')[0]
    video_webm = videoname + '.webm'
    command = 'ffmpeg -y -i  ' + video_mp4 +  ' -f webm -c:v libvpx -b:v 1M -acodec libvorbis ' + video_webm + ' -hide_banner'
    subprocess.call(command, shell=True)

#-------------------------------------------------------------------------------
def convert_mp4_to_webm_rt(video_mp4):
    videoname = video_mp4.split('.mp4')[0]
    video_webm = videoname + '.webm'
    command = 'ffmpeg -y -i  ' + video_mp4 +  ' -f webm -c:v libvpx -b:v 1M -acodec libvorbis ' + video_webm + ' -hide_banner'
    subprocess.call(command, shell=True)
    return(video_webm)

#--------------------------------------------------------------------------------
def convert_mp4_to_webm_small (video_mp4):
    #reduces filesize; 640 wide output, some loss of quality
    videoname = video_mp4.split('.mp4')[0]
    video_webm = videoname + '.webm'
    command = 'ffmpeg -y -i ' + video_mp4 + ' -c:v libvpx-vp9 -b:v 0.33M -c:a libopus -b:a 96k -filter:v scale=640:-1 ' + video_webm + ' -hide_banner'
    subprocess.call(command, shell=True)

#-------------------------------------------------------------------------------
def chunk_large_videofile(video, chunksize, location):
    #convert minutes to seconds
    seg = seconds_to_hms((chunksize*60))
    name = video.split('.')[0]
    format = video.split('.')[1]
    command = 'ffmpeg -y -i ' + video + ' -c copy -map 0 -segment_time ' + seg + ' -f segment -reset_timestamps 1 ' + name + '_' + '%02d.' + format
    subprocess.call(command, shell=True)
    path, dirs, files = next(os.walk(location))
    #one file is the orginal
    nfiles = len(files)-1

    return(nfiles)

#------------------------------------------------------------------------------
def cleanrecording(audiofile):
    noiseoutput = 'bnoise_'+ audiofile
    cleanoutput = 'clean_' + audiofile

    magic = 0.1                 #0.2 .... range(0.0 - 1.0)
    stime = '00:00:00.75'       #the first 0.75 seconds

    command = 'ffmpeg -loglevel panic -y -ss 00:00:00 -t ' + stime + ' -i ' + audiofile + ' -acodec copy ' + noiseoutput
    subprocess.call(command, shell=True)
    #generate a noise profile with sox
    command = 'sox ' + noiseoutput + ' -n noiseprof noise.prof'
    #this is blocking, so dont have to check status
    subprocess.call(command, shell=True)
    #remove that noise profile  from the inputaudio
    command = 'sox ' + audiofile +  ' ' + cleanoutput + ' noisered noise.prof ' + str(magic)
    subprocess.call(command, shell=True)
    return(cleanoutput)

#-------------------------------------------------------------------------------
def get_segment(videofile, start_cut, end_cut):
    s = seconds_to_hms(start_cut)
    l = end_cut - start_cut
    d = seconds_to_hms(l)
    cut = 'cut_' + videofile

    command = 'ffmpeg -loglevel panic -y -ss ' + str(s) + ' -i ' + videofile   + ' -to ' + str(d) + ' -c copy -an ' + cut
    subprocess.call(command, shell=True)
    return(cut)

#-------------------------------------------------------------------------------
def voiceover_recording(dur, card, device, output):
    command = 'ffmpeg -loglevel panic -y -f alsa -ac 1 -i plughw:' + str(card) + ',' + str(device) + ' -t ' + str(dur) + ' ' +  output
    subprocess.call(command, shell=True)

#-------------------------------------------------------------------------------
def record_and_playback(dur, card, device, output):
    command1 = 'ffmpeg -loglevel panic -y -f alsa -ac 1 -i plughw:' + str(card) + ',' + str(device) + ' -t ' + str(dur) + ' ' +  output
    subprocess.call(command1, shell=True)
    command2 = "aplay " + output
    subprocess.call(command2, shell=True)

#-------------------------------------------------------------------------------
def combine_recordingvideo(audiofile, videofile, output):
    command = 'ffmpeg -y -i ' + videofile + ' -i ' + audiofile + ' -map 0:0 -map 1:0 -c:v copy -c:a copy -c:a aac -b:a 256k -shortest ' + output
    print(command)
    subprocess.call(command, shell=True)
    return('combo complete')
    '''
    print('convert mkv to mp4')
    #command = 'ffmpeg -loglevel panic -y -i ' + fin_mkv + ' -c:v libx264 -preset slow -crf 21 ' +  fin_mp4
    #command = 'ffmpeg -loglevel panic -y -i ' + fin_mkv + ' -c copy' +  fin_mp4
    subprocess.call(command, shell=True)
    '''

#-------------------------------------------------------------------------------
def create_images_from_video(savepath, category, videonamepath, framerate):
    tempok = os.path.isdir(savepath)
    if(tempok):
        pass
    else:
        os.mkdir(savepath)

    start = str(0)
    end_set = " -f image2 "
    out = ' -start_number ' + start + ' ' + savepath + '%04d.jpg'
    s1 = 'ffmpeg -loglevel panic -y -i '

    command = s1 + videonamepath + ' -r ' + str(framerate) + end_set + out
    subprocess.call(command, shell=True)

    path, dirs, files = next(os.walk(savepath))
    files = files.sort(key=lambda f: int(re.sub('\D', '', f)))

#-------------------------------------------------------------------------------
