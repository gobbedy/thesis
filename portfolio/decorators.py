from functools import wraps
from time import time
from inspect import getouterframes, currentframe
import logging
from line_profiler import LineProfiler

# credit: https://stackoverflow.com/a/27737385/8112889
def timed(f):
    @wraps(f)
    def wrap(self, *args, **kw):
        
        # set logger indentation according to nesting of function
        level = len(getouterframes(currentframe()))
        indent = level * ' '
        formatter = logging.Formatter(indent + '%(name)s - %(levelname)s: - %(message)s')
        self.ch.setFormatter(formatter)
        
        # print start of function info
        self.logger.info('**************************************************')
        self.logger.info('Started ' + f.__name__)
        
        # time function and call it
        ts = time()
        result = f(self, *args, **kw)
        te = time()
        
        # print end of function info
        self.logger.info('Finished ' + f.__name__ + ': took %2.4f seconds.' % (te-ts))
        self.logger.info('**************************************************')
        
        # return function's result
        return result
    return wrap


def profile(f):

    @wraps(f)
    def wrap(self, *args, **kw):

        # profile function and call it
        lp = LineProfiler()
        lp_wrapper = lp(f)
        result = lp_wrapper(self, *args, **kw)
        lp.print_stats()

        # return function's result
        return result
    return wrap

