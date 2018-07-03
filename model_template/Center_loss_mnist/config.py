# coding=utf-8
from common import *

class Config(object):
    '''
    Config类初始化的时候会自动创建一个以model name为名称的文件夹
    当调用update_time_stamp方法时，会在model name文件夹下生成以时间为开头的子文件夹，其中
    有log文件夹和snapshots文件夹，用于存放训练过程数据以及保存模型
    Demo:
    >>> config = Config
    /data/checkpoints/center_loss/center_loss
    >>> config.update_time_stamp()
    /data/checkpoints/center_loss/center_loss|
                                             |2018-06-25-14-45-50
                                                                |log
                                                                |snapshots

    '''
    model_name  = 'center_loss'
    data_dir   = '/data/mnist'
    model_dir  = '/data/checkpoints/center_loss'
    num_classes = 10
    image_shape = (28,28)
    num_epochs  = 10
    alpha       = 0.5
    beta        = 0.5
    center_dim  = 2
    lr          = 1e-2

    def __init__(self):
        self.update_time_stamp()
        pass

    def update_time_stamp(self):
        '''
        获取当前时间并更新time stamp，同时创建以time stamp为开头的文件夹用于
        存放log和snapshots
        Demo：
        >>> config.update_time_stamp()
        /data/checkpoints/center_loss/center_loss|
                                                 |2018-06-25-14-45-50
                                                                    |log
                                                                    |snapshots
        '''
        self.time_stamp = get_time_string()
        self.set_path()

    def set_path(self):
        '''
        Will be called in self.update_time_stamp
        获取time stamp并且生成对应的文件夹
        '''
        time_stamp = self.time_stamp
        # root_save_path: '/data/ckeckpoints/center_loss/2018-06-25-10-30-56'
        root_dir = join(self.model_dir, self.model_name, self.time_stamp)
        log_dir = join(root_dir, 'log')
        log_path = join(log_dir, 'log.txt')
        ckpt_dir = join(root_dir, 'snapshots')
        ckpt_path = join(ckpt_dir, 'net.ckpt')

        make_dir(root_dir)
        make_dir(log_dir)
        make_dir(ckpt_dir)
        self.root_dir = root_dir
        self.log_dir = log_dir
        self.log_path = log_path

        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        pass

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
