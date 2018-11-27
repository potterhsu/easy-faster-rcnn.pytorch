import logging


class Logger(object):
    Initialized = False

    @staticmethod
    def initialize(path_to_log_file):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(path_to_log_file),
                                      logging.StreamHandler()])
        Logger.Initialized = True

    @staticmethod
    def log(level, message):
        assert Logger.Initialized, 'Logger has not been initialized'
        logging.log(level, message)

    @staticmethod
    def d(message):
        Logger.log(logging.DEBUG, message)

    @staticmethod
    def i(message):
        Logger.log(logging.INFO, message)

    @staticmethod
    def w(message):
        Logger.log(logging.WARNING, message)

    @staticmethod
    def e(message):
        Logger.log(logging.ERROR, message)
