# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:26:04 2019

@author: mshem
"""
from glob import glob

from numpy import savetxt
from numpy import array as ar
from numpy import genfromtxt

from os.path import dirname
from os.path import basename
from os.path import realpath
from os.path import splitext
from os.path import abspath

from datetime import datetime

from getpass import getuser

from platform import uname
from platform import system

from inspect import currentframe
from inspect import getouterframes
    
class Logger():
    """Logger Class"""
    def __init__(self, blabla=True, debugger=None):
        self.blabla = blabla
        self.log_file = debugger
        if self.log_file is None:
            self.log("LOGGER Started with no log file")
        else:
            self.log("LOGGER Started with log file at {}".format(
                    self.log_file))
        
    def time_stamp(self, seperators="-T:"):
        """Returns timestamp"""
        FORMAT = "%Y{0}%m{0}%d{1}%H{2}%M{2}%S".format(
                seperators[0], seperators[1], seperators[2])
        return(str(datetime.utcnow().strftime(FORMAT)))
    
    def system_info(self):
        """Returns System Information"""
        si = uname()
        return("{}, {}, {}".format(si[0], si[2], si[5]))
        
    def caller_function(self, pri=False):
        """Returns Caller function of a function"""
        curframe = currentframe()
        calframe = getouterframes(curframe, 2)
        caller = calframe
        self.system_info()
        if pri:
            return("{}>{}>{}".format(
                    caller[0][3], caller[1][3], caller[2][3]))
        else:
            return(caller)
            
    def print_if(self, text):
        """Prints text if verb is True"""
        curframe = currentframe()
        calframe = getouterframes(curframe, 2)
        caller = calframe
        the_caller = ""
        for i in caller:
            the_caller = "{}<-{}({})".format(the_caller, basename(i.filename), i.lineno)
        if self.blabla:
            print("[{}|{}][{}] --> {}".format(
                    self.time_stamp(), self.system_info(),
                    the_caller[2:], text))
            
    def log(self, text):
        """Logs text if debugger is a path"""
        self.print_if(text)
        if self.log_file is not None:
            lf = abspath(self.log_file)
            try:
                log_file = open(lf, "a")
                log_file.write("Time: {}\n".format(self.time_stamp()))
                log_file.write("System Info: {}\n".format(self.system_info()))
                log_file.write("Log: {}\n".format(text))
                log_file.write("Function: {}\n\n".format(
                        self.caller_function()))
                log_file.close()
            except Exception as e:
                print(e)
        
    def is_it_windows(self):
        """Checks if system is windows"""
        self.log("Checking if the OS is Windows")
        return(system() == 'Windows')
        
    def is_it_linux(self):
        """Checks if system is GNU/Linux"""
        self.log("Checking if the OS is Linux")
        return(system() == 'Linux')
        
    def is_it_other(self):
        """Checks if system is neither Windows nor GNU/Linux """
        self.log("Checking if the OS is Other")
        return(not (self.is_it_linux() or self.is_it_windows()))
        
class File:
    """File Operation Class"""
    def __init__(self, logger):
        self.logger = logger
        
    def list_in_path(self, path):
        """Returns file/directory in a given path"""
        try:
            pt = self.abs_path(path)
            return(sorted(glob(pt)))
        except Exception as e:
            self.logger.log(e)
        
    def abs_path(self, path):
        """Return Absolute path of a given path"""
        try:
            return(abspath(path))
        except Exception as e:
            self.logger.log(e)
            
    def get_base_name(self, src):
        """Returns base path of a given file"""
        self.logger.log("Finding path and file name for {0}".format(src))
        try:
            pn = dirname(realpath(src))
            fn = basename(realpath(src))
            return(pn, fn)
        except Exception as e:
            self.logger.log(e)
            
    def get_extension(self, src):
        """Returns extension of a given file"""
        self.logger.log("Finding extension for {0}".format(src))
        try:
            return(splitext(src))
        except Exception as e:
            self.logger.log(e)
            
    def split_file_name(self, src):
        """Retuns path and name of a file"""
        self.logger.log("Chopping path {0}".format(src))
        try:
            path, name = self.get_base_name(src)
            name, extension = self.get_extension(name)
            return(path, name, extension)
        except Exception as e:
            self.logger.log(e)
            
    def save_numpy(self, src, arr, dm=" ", h=""):
        """Writes an array to a file"""
        self.logger.log("Writing to {0}".format(src))
        try:
            arr = ar(arr)
            savetxt(src, arr, delimiter=dm, newline='\n', header=h)
        except Exception as e:
            self.logger.log(e)
            
    def read_array(self, src, dm=" ", dtype=float, reshape=False):
        """Reads an array from a file"""
        self.logger.log("Reading {0}".format(src))
        try:
            d = genfromtxt(src, comments='#', delimiter=dm, dtype=dtype)
            if reshape:
                return(d.reshape(-1, 1))
            else:
                return(d)
        except Exception as e:
            self.logger.log(e)
    