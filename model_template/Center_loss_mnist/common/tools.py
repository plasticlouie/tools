# coding=utf-8
import numpy as np
import scipy.io
import time
import os
from os.path import join
import shutil

def move_file(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print "move %s -> %s"%( srcfile,dstfile)

def copy_file(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print "copy %s -> %s"%( srcfile,dstfile)

#
# Time
#
def get_date_string():
    '''
    Demo:
    >>> date_str = get_date_string()
    >>> print date_str
    2018-06-12-09-44-50
    '''
    date_str = time.strftime('%Y-%m-%d',time.localtime())
    return date_str

def get_time_string():
    '''
    Demo:
    >>> time_str = get_time_string()
    >>> print time_str
    2018-06-12-09-44-50
    '''
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
    return time_str

def make_dir(path):
    '''
    Demo:
    >>> make_dir('/data/mnist')
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def write_line(log_path, string, overwrite=False, newline=True):
    '''
    Demo1:
    >>> write_line('./log.txt', 'line1')
    >>> write_line('./log.txt', 'line2')
    '''
    if overwrite:
        f = open(log_path, 'w')
    else:
        f = open(log_path, 'a')
    if newline:
        string += '\n'
    f.write(string)
    f.close()


if __name__ == '__main__':
    time_str = get_time_string()
    print time_str

    for i in range(10):
        write_line('./test.log','line'+str(i))
