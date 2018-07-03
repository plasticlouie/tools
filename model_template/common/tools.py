# coding=utf-8
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

class Log_collector(object):
    def __init__(self, log_dir = './log'):
        self.log_dict = {}
        self.log_dir = log_dir
        pass

    def add_log(self, name, value):
        if name in log_dict.keys():
            log_dict[name] += self.to_list(value)
        else:
            log_dict[name] = self.to_list(value)

    def to_list(self, value):
        if type(value)==list:
            return value
        if type(value)==np.ndarray:
            return np.ndarray.tolist(value)
        if type(value) in [int, float]:
            return [value]

    def display_log(self):
        pass

    def write_log(self, combine_list=None):
        for key in self.log_dict.keys():
            self.write_txt(key, log_dict[key])

        if combine_list is not None:
            value_list = []
            for i in range(len(combine_list)):
                value_list = value_list + self.log_dict[combine_list[i]]
            self.write_txt(combine_list, value_list)
        pass

    def write_txt(self, log_name, value_list, log_dir=None, overwrite=False):
        '''
        Demo1:
        >>> write_line('./log.txt', 'line1')
        >>> write_line('./log.txt', 'line2')
        '''
        if log_dir is None:
            log_dir = self.log_dir
        log_path = join(log_dir, log_name+'.txt')

        if overwrite:
            f = open(log_path, 'w')
        else:
            f = open(log_path, 'a')

        if type(log_name) is not list:
            log_name = [log_name]
            value_list = [value_list]

        for i in range(len(value_list[0])):
            string = str(i)
            for j in range(len(log_name)):
                value = value_list[i][j]
                value_str = '{:8}'.format(value)
                string = string + " " + log_name[i] + ":" + value_str
            string += '\n'
            f.write(string)
        f.close()

if __name__ == '__main__':
    time_str = get_time_string()
    print time_str

    for i in range(10):
        write_line('./test.log','line'+str(i))
