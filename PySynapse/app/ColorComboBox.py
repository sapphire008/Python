# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:48:23 2018

Color drop-down combobox

@author: Edward

"""
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from pdb import set_trace

class ColorDropDownCombobox(QtWidgets.QComboBox):
    def __init__(self, parent=None,
                 colors=('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf'), default=None):
        super(ColorDropDownCombobox, self).__init__(parent)
        self.colors = list(colors)
        self.setColors(self.colors)
        self.setEditable(True)
        self.lineEdit().setMaxLength(45)
        if default is not None:
            self.lineEdit().setText(default)

    def setColors(self, colors=None):
        if colors is None: return
        if isinstance(colors, (tuple, list, np.ndarray)):
            self.colors = list(colors)
        for n, c in enumerate(self.colors):
            myqcolor = self.parseColor(c)
            self.insertItem(n, c)
            self.setItemData(n, myqcolor, role=QtCore.Qt.DecorationRole)

    def setColorAt(self, color=None, index=None):
        if color is not None and index is not None:
            self.colors[index] = color
            self.setItemData(index, self.parseColor(color), role=QtCore.Qt.DecorationRole)

    @staticmethod
    def parseColor(c):
        if isinstance(c, str):
            if c[0] == '#':  # hex to rgb
                c = tuple(int(c.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
                return QtGui.QColor(*c, alpha=255)
            elif QtGui.QColor.isValidColor(c):  # test if it is valid QColor
                return QtGui.QColor(c)
        elif isinstance(c, (tuple, list, np.ndarray)) and len(c) == 3:  # rgb
            return QtGui.QColor(*c, alpha=255)
        else:
            raise (TypeError("Unrecognized type of 'colors' input"))




