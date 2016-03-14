'''
This file is used to report the amount of time elapsed, and how much remains.

Used like this::

    progress = ProgressReporter()
    progress.start()
    for ...:
        ... do some work ...
        progress.update(percentcomplete)
    progress.finish()

It will give a progress report every 60 seconds by default.
    
Adapted and simplified from progressreporting.py in the Brian neural network
simulator (http://briansimulator.org).
'''
import sys, time
from kwiklib import info

__all__ = ['ProgressReporter']

def time_rep(t):
    '''
    Textual representation of time t given in seconds
    '''
    t = int(t)
    if t < 60:
        return str(t) + 's'
    secs = t % 60
    mins = t // 60
    if mins < 60:
        return str(mins) + 'm ' + str(secs) + 's'
    mins = mins % 60
    hours = t // (60 * 60)
    if hours < 24:
        return str(hours) + 'h ' + str(mins) + 'm ' + str(secs) + 's'
    hours = hours % 24
    days = t // (60 * 60 * 24)
    return str(days) + 'd ' + str(hours) + 'h ' + str(mins) + 'm ' + str(secs) + 's'

def make_text_report(elapsed, complete):
    s = str(int(100 * complete)) + '% complete, '
    s += time_rep(elapsed) + ' elapsed'
    if complete > .001:
        remtime = elapsed / complete - elapsed
        s += ', ' + time_rep(remtime) + ' remaining.'
    else:
        s += '.'
    return s

class ProgressReporter(object):
    '''
    Standard text progress reports
    
    Initialised with an optional argument:
    
    ``period``
        How often reports should be generated in seconds.
    
    Methods:
    
    .. method:: start()
    
        Call at the beginning of a task to start timing it.
    
    .. method:: finish()
    
        Call at the end of a task to finish timing it.
    
    .. method:: update(complete[, extrainfo])
    
        Call with the fraction of the task completed, between 0 and 1, and
        the optional extrainfo parameters is a string giving extra information
        about the progress.        
    '''
    def __init__(self, period=60.0):
        self.period = float(period)
        self.start()

    def start(self):
        self.start_time = time.time()
        self.next_report_time = self.start_time

    def finish(self):
        self.update(1)

    def update(self, complete, extrainfo=''):
        cur_time = time.time()
        if (cur_time > self.next_report_time) or \
                complete == 1.0 or complete == 1:
            self.next_report_time = cur_time + self.period
            elapsed = time.time() - self.start_time
            s = make_text_report(elapsed, complete)
            if extrainfo:
                s += ' '+extrainfo
            info(s)
