import numpy as np
import scipy.io
import os
from os.path import join
import shutil

class Log_collector(object):
    def __init__(self, log_dir = './log'):
        # Log dictionary
        self.log_dict = {}
        self.log_type = {}
        # Log saving directory
        self.log_dir = log_dir

    def new_log(self, name, type):
        pass

    def add_log(self, name, value):
        '''

        '''
        if name in self.log_dict.keys():
            self.log_dict[name] += self.to_list(value)
        else:
            self.log_dict[name] = self.to_list(value)

    def clear_log(self, names=None):
        if names is None:
            self.log_dict = {}
            return
        names = self.to_list(names)
        for key in names:
            if key in self.log_dict:
                self.log_dict.pop(key)
            else:
                print("log dictionary has no key: "+name)

    def to_list(self, value):
        '''
        Convert data to list
        '''
        if type(value)==list:
            return value
        if type(value)==np.ndarray:
            return np.ndarray.tolist(value)
        return [value]

    def display_log(self):
        for key in self.log_dict.keys():
            print(key+': '+str(type(self.log_dict[key][0])))
            #
            print(key+': ('+str(len(self.log_dict[key]))+',)')

    def write_image_log(self, log_names):
        pass

    def write_log(self, combine_list=None):
        for key in self.log_dict.keys():
            self.write_txt(key, self.log_dict[key])
            self.write_mat(key, self.log_dict[key])

        if combine_list is not None:
            value_list = []
            for i in range(len(combine_list)):
                value_list = value_list + [self.log_dict[combine_list[i]]]
            self.write_txt(combine_list, value_list)
        pass

    def write_mat(self, log_name, value):
        '''
        Will be called in self.write_log
        '''
        log_path    = join(self.log_dir, log_name+'.mat')
        # value       = np.asarray(value)
        mat_dict    = {log_name: value}
        scipy.io.savemat(log_path, mat_dict)

    def write_txt(self, log_name, value_list, overwrite=True):
        '''
        Will be called in self.write_log
        '''

        if type(log_name) is not list:
            log_path = join(self.log_dir, log_name+'.txt')
        else:
            log_path = join(self.log_dir, 'log.txt')

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
                value = value_list[j][i]
                if type(value) == str:
                    value_str = value
                else:
                    value_str = ('{:8}'.format(value)).lstrip()
                string = string + " " + log_name[j] + ":" + value_str
            string += '\n'
            f.write(string)
        f.close()
