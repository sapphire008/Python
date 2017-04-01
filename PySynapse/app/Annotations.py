# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 19:40:31 2016

Adding annotations to display as well as export

@author: Edward
"""

import sys
import os
import fileinput
from PyQt4 import QtCore, QtGui

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

sys.path.append(os.path.join(__location__, '..')) # for debug only
from util.MATLAB import *


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class AnnotationSetting(QtGui.QDialog):
    def __init__(self, parent=None):
        super(AnnotationSetting, self).__init__(parent)
        self.setWindowIcon(QtGui.QIcon('resources/icons/setting.png'))
        self.isclosed = False
        self.parent = parent
        self.initialTypeSelectionDialog() # type of annotation setting to make
        self.artists = dict()
        self.setLayout(QtGui.QVBoxLayout())

        # Call the corresponding setting windows to get annotation object properties
        if self.type == 'box':
            self.setWindowTitle("Box Annotations")
            widgetFrame = self.boxSettings()
        else:
            raise(NotImplementedError("'{}' annotation object has not been implemented yet".format(self.type)))
        
        # buttons for saving the settings and exiting the settings window
        OK_button = QtGui.QPushButton('OK')
        OK_button.setDefault(True)
        OK_button.clicked.connect(lambda: self.updateSettings(closeWidget=True))
        Cancel_button = QtGui.QPushButton('Cancel')
        Cancel_button.clicked.connect(self.close)
        self.buttonGroup = QtGui.QGroupBox()
        self.buttonGroup.setLayout(QtGui.QHBoxLayout())
        self.buttonGroup.layout().addWidget(OK_button, 0)
        self.buttonGroup.layout().addWidget(Cancel_button, 0)

        self.layout().addWidget(widgetFrame)
        self.layout().addWidget(self.buttonGroup)

    def initialTypeSelectionDialog(self):
        ann_obj = ['box',  # [x1, y1, x2, y2, linewidth, linestyle, color]
                   'line',  # [x1, y1, x2, y2, linewidth, linestyle, color]
                   'circle',  # [center_x, center_y, a, b, rotation, linewidth, linestyle, color]
                   'arrow',  # [x, y, x_arrow, y_arrow, linewidth, linestyle, color]
                   'symbol']  # ['symbol', x, y, markersize, color]

        selected_item, ok = QtGui.QInputDialog.getItem(self, "Select annotation object type",
                                              "annotation objects", ann_obj, 0, False)
        self.type = selected_item

    def boxSettings(self):
        # return a dictionary of the box annotation artist
        widgetFrame = QtGui.QFrame()
        widgetFrame.setLayout(QtGui.QGridLayout())
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        widgetFrame.setSizePolicy(sizePolicy)
        widgetFrame.setObjectName(_fromUtf8("boxSettingsWidgetFrame"))

        # Settings for a box
        x0_label = QtGui.QLabel('X0')
        x0_text = QtGui.QLineEdit('0')
        y0_label = QtGui.QLabel('Y0')
        y0_text = QtGui.QLineEdit('0')
        w_label = QtGui.QLabel('Width')
        w_text = QtGui.QLineEdit('10')
        h_label = QtGui.QLabel('Height')
        h_text = QtGui.QLineEdit('10')
        line_checkbox = QtGui.QCheckBox('Line')
        line_checkbox.setCheckState(2)
        lw_label = QtGui.QLabel('Line Width')
        lw_text = QtGui.QLineEdit('0.5669291338582677')
        ls_label = QtGui.QLabel('Line Style')
        ls_text = QtGui.QLineEdit('-')
        lc_label = QtGui.QLabel('Line Color')
        lc_text = QtGui.QLineEdit('k') # single letters, or triplet in brackets [1,0,0]
        fill_checkbox = QtGui.QCheckBox('Fill')
        fill_checkbox.setCheckState(0)
        fc_label = QtGui.QLabel('Fill Color')
        fc_text = QtGui.QLineEdit('w') # single letters, or triplet in brackets [1,0,0]
        fa_label = QtGui.QLabel('Fill Alpha')
        fa_text = QtGui.QLineEdit('100')
        fa_suffix_label = QtGui.QLabel('%')

        # Make a dictionary of the values
        self.settingDict = dict()
        self.settingDict['x0'] = x0_text
        self.settingDict['y0'] = y0_text
        self.settingDict['width'] = w_text
        self.settingDict['height'] = h_text
        self.settingDict['line'] = line_checkbox
        self.settingDict['linewidth'] = lw_text
        self.settingDict['linestyle'] = ls_text
        self.settingDict['linecolor'] = lc_text
        self.settingDict['fill'] = fill_checkbox
        self.settingDict['fillcolor'] = fc_text
        self.settingDict['fillalpha'] = fa_text

        # Add the widgets to the window
        widgetFrame.layout().addWidget(x0_label, 0, 0, 1, 1)
        widgetFrame.layout().addWidget(x0_text, 0, 1, 1, 1)
        widgetFrame.layout().addWidget(y0_label, 0, 2, 1, 1)
        widgetFrame.layout().addWidget(y0_text, 0, 3, 1, 1)
        widgetFrame.layout().addWidget(w_label, 1, 0, 1, 1)
        widgetFrame.layout().addWidget(w_text, 1, 1, 1, 1)
        widgetFrame.layout().addWidget(h_label, 1, 2, 1, 1)
        widgetFrame.layout().addWidget(h_text, 1, 3, 1, 1)
        widgetFrame.layout().addWidget(line_checkbox, 2, 0, 1, 2)
        widgetFrame.layout().addWidget(lw_label, 2, 2, 1, 1)
        widgetFrame.layout().addWidget(lw_text, 2, 3, 1, 1)
        widgetFrame.layout().addWidget(ls_label, 3, 0, 1, 1)
        widgetFrame.layout().addWidget(ls_text, 3, 1, 1, 1)
        widgetFrame.layout().addWidget(lc_label, 3, 2, 1, 1)
        widgetFrame.layout().addWidget(lc_text, 3, 3, 1, 1)

        widgetFrame.layout().addWidget(fill_checkbox, 4, 0, 1, 2)
        widgetFrame.layout().addWidget(fc_label, 4, 2, 1, 1)
        widgetFrame.layout().addWidget(fc_text, 4, 3, 1, 1)
        widgetFrame.layout().addWidget(fa_label, 5, 0, 1, 1)
        widgetFrame.layout().addWidget(fa_text, 5, 1, 1, 1)
        widgetFrame.layout().addWidget(fa_suffix_label, 5, 2, 1, 1)

        return widgetFrame

    def checkSettingUpdates(self):
        if self.type == 'box':
            keys = ['x0', 'y0', 'width', 'height', 'linewidth', 'linestyle', 'linecolor', 'fillcolor', 'fillalpha']
            for k in keys:
                if self.artists[k] == '':
                    msg = QtGui.QMessageBox()
                    msg.setWindowTitle("Error")
                    msg.setText("'{}' argument cannot be empty when drawing a '{}'".format(k, self.type))
                    msg.exec_()
                    return False

            return True

    def updateSettings(self, closeWidget=False):
        for k, v in self.settingDict.items():
            if isinstance(v, QtGui.QComboBox):
                val = v.currentText()
            elif isinstance(v, QtGui.QLineEdit):
                val = v.text()
            elif isinstance(v, QtGui.QCheckBox):
                val = True if v.checkState() > 0 else False
            elif isinstance(v, QtGui.QSpinBox):
                val = v.value()
            else:
                raise (TypeError('Unrecognized type of setting item'))

            self.artists[k] = val

        # sanity check
        state = self.checkSettingUpdates()
        if not state:
            return

        if closeWidget:
            self.accept()



if __name__ == '__main__':
    #    iniPath = 'D:/Edward/Documents/Assignments/Scripts/Python/PySynapse/resources/config.ini'
    #    with fileinput.input(iniPath, inplace=True, backup='.bak') as f:
    #        for line in f:
    #            if line[0] == '#':
    #                print('#asdf.mat')
    #            else:
    #                print(line, end='')

    app = QtGui.QApplication(sys.argv)
    ex = AnnotationSetting()
    ex.show()
    if ex.exec_():
        print(ex.artists)
    # fff = app.exec_()
    # print(ex.artists)
    # sys.exit(fff)

