#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt4 import QtGui
import sys
import pyperclip as Pclip

def main():
    app = QtGui.QApplication(sys.argv)
    fname = QtGui.QFileDialog.getExistingDirectory(None, 'Open folder', '/home')
    if fname:
        Pclip.copy(fname) # copy selected folder to clipboard

if __name__ == "__main__":
    main()