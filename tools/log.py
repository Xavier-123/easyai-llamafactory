import logging
import os
from logging import handlers


class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='W0', backCount=10,
                 fmt='[%(asctime)s]-[%(pathname)s:%(lineno)d]-[%(levelname)s] %(message)s'):
        self.logger = logging.getLogger(filename)
        # 设置日志格式
        format_str = logging.Formatter(fmt)
        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))
        # 往屏幕上输出
        sh = logging.StreamHandler()
        # 设置屏幕上显示的格式
        sh.setFormatter(format_str)
        # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，
        # when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when,
                                               backupCount=backCount, encoding='utf-8')

        # 设置文件里写入的格式
        th.setFormatter(format_str)

        # 把对象加到logger里
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


curr_path = os.path.split(os.path.abspath(__file__))[0]

proj_path = os.path.dirname(os.path.dirname(curr_path))
log_path = proj_path + '/logs'
print(log_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logger = Logger(log_path + '/api.log', level='info').logger
