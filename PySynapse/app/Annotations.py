# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 19:40:31 2016

Adding annotations to display as well as export

@author: Edward
"""

import sys
import os
import fileinput
from PyQt5 import QtCore, QtGui, QtWidgets
from app.ColorComboBox import ColorDropDownCombobox

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

sys.path.append(os.path.join(__location__, '..')) # for debug only
from util.MATLAB import *

from collections import OrderedDict


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig)

class AnnotationSetting(QtWidgets.QDialog):
    # Class variable
    ann_obj = ['box',  # [x1, y1, x2, y2, linewidth, linestyle, color]
               'line',  # [x1, y1, x2, y2, linewidth, linestyle, color]
               'circle',  # [center_x, center_y, a, b, rotation, linewidth, linestyle, color]
               'arrow',  # [x, y, x_arrow, y_arrow, linewidth, linestyle, color]
               'symbol',  # ['symbol', x, y, markersize, color]
               'ttl']  # TTL triggered stimulus [bool_convert_pulse_to_step]
    def __init__(self, parent=None, artist=None):
        super(AnnotationSetting, self).__init__(parent)
        self.setWindowIcon(QtGui.QIcon('resources/icons/setting.png'))
        self.isclosed = False
        self.parent = parent
        if artist is None:
            self.initialTypeSelectionDialog() # type of annotation setting to make
        else:
            self.type = artist['type']
        self.artist = dict() if artist is None else artist
        self.settingDict = dict()
        self.setLayout(QtWidgets.QVBoxLayout())

        # Call the corresponding setting windows to get annotation object properties
        if self.type == 'box':
            self.setWindowTitle("Box Annotations")
            widgetFrame = self.boxSettings()
        elif self.type == 'line':
            self.setWindowTitle('Line Annotation')
            widgetFrame = self.lineSettings()
        elif self.type == 'ttl':
            self.setWindowTitle("TTL Annotation")
            widgetFrame = self.ttlSettings()
        else:
            raise(NotImplementedError("'{}' annotation object has not been implemented yet".format(self.type)))
        
        # buttons for saving the settings and exiting the settings window
        OK_button = QtWidgets.QPushButton('OK')
        OK_button.setDefault(True)
        OK_button.clicked.connect(lambda: self.updateSettings(closeWidget=True))
        Cancel_button = QtWidgets.QPushButton('Cancel')
        Cancel_button.clicked.connect(self.close)
        self.buttonGroup = QtWidgets.QGroupBox()
        self.buttonGroup.setLayout(QtWidgets.QHBoxLayout())
        self.buttonGroup.layout().addWidget(OK_button, 0)
        self.buttonGroup.layout().addWidget(Cancel_button, 0)

        self.layout().addWidget(widgetFrame)
        self.layout().addWidget(self.buttonGroup)

    def initialTypeSelectionDialog(self, current_index=0):
        selected_item, ok = QtWidgets.QInputDialog.getItem(self, "Select annotation object type",
                                              "annotation objects", self.ann_obj, current_index, False)
        self.type = selected_item

    def boxSettings(self):
        """return a dictionary of the box annotation artist"""
        widgetFrame = QtWidgets.QFrame()
        widgetFrame.setLayout(QtWidgets.QGridLayout())
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        widgetFrame.setSizePolicy(sizePolicy)
        widgetFrame.setObjectName(_fromUtf8("boxSettingsWidgetFrame"))

        # Settings for a box
        x0_label = QtWidgets.QLabel('X0')
        x0_text = QtWidgets.QLineEdit(self.parseArtist(field='x0', default='0', return_type=str))
        y0_label = QtWidgets.QLabel('Y0')
        y0_text = QtWidgets.QLineEdit(self.parseArtist(field='y0', default='0', return_type=str))
        w_label = QtWidgets.QLabel('Width')
        w_text = QtWidgets.QLineEdit(self.parseArtist(field='width', default='500', return_type=str))
        h_label = QtWidgets.QLabel('Height')
        h_text = QtWidgets.QLineEdit(self.parseArtist(field='height', default='10', return_type=str))
        line_checkbox = QtWidgets.QCheckBox('Line')
        line_checkbox.setCheckState(self.parseArtist(field='line', default=2, return_type=bool))
        lw_label = QtWidgets.QLabel('Line Width')
        lw_text = QtWidgets.QLineEdit(self.parseArtist(field='linewidth', default='0.5669291338582677', return_type=str))
        ls_label = QtWidgets.QLabel('Line Style')
        ls_text = QtWidgets.QLineEdit(self.parseArtist(field='linestyle', default='-', return_type=str))
        ls_text.setToolTip('"-" (default), "--", "-.", ":"')
        lc_label = QtWidgets.QLabel('Line Color')
        lc_text = ColorDropDownCombobox(default=self.parseArtist(field='linecolor', default='k', return_type=str))
        #lc_text = QtWidgets.QLineEdit(self.parseArtist(field='linecolor', default='k', return_type=str)) # single letters or hex string
        lc_text.setToolTip('Single letter or hex value of the color')
        fill_checkbox = QtWidgets.QCheckBox('Fill')
        fill_checkbox.setCheckState(self.parseArtist(field='fill', default=0, return_type=bool))
        fc_label = QtWidgets.QLabel('Fill Color')
        fc_text = ColorDropDownCombobox(default=self.parseArtist(field='fillcolor', default='#1f77b4', return_type=str))
        # fc_text = QtWidgets.QLineEdit(self.parseArtist(field='fillcolor', default='#1f77b4', return_type=str))
        fc_text.setToolTip('Single letter or hex value of the color')
        fa_label = QtWidgets.QLabel('Fill Alpha')
        fa_text = QtWidgets.QLineEdit(self.parseArtist(field='fillalpha', default='100', return_type=str))
        fa_suffix_label = QtWidgets.QLabel('%')

        # Make a dictionary of the values
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

    def lineSettings(self):
        widgetFrame = QtWidgets.QFrame()
        widgetFrame.setLayout(QtWidgets.QGridLayout())
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        widgetFrame.setSizePolicy(sizePolicy)
        widgetFrame.setObjectName(_fromUtf8("lineSettingsWidgetFrame"))

        # Settings for a line
        x0_label = QtWidgets.QLabel('X0')
        x0_text = QtWidgets.QLineEdit(self.parseArtist(field='x0', default='0', return_type=str))
        y0_label = QtWidgets.QLabel('Y0')
        y0_text = QtWidgets.QLineEdit(self.parseArtist(field='y0', default='0', return_type=str))
        x1_label = QtWidgets.QLabel('X1')
        x1_text = QtWidgets.QLineEdit(self.parseArtist(field='x1', default='1000', return_type=str))
        y1_label = QtWidgets.QLabel('Y1')
        y1_text = QtWidgets.QLineEdit(self.parseArtist(field='y1', default='0', return_type=str))

        lw_label = QtWidgets.QLabel('Line Width')
        lw_text = QtWidgets.QLineEdit(self.parseArtist(field='linewidth', default='0.5669291338582677', return_type=str))
        ls_label = QtWidgets.QLabel('Line Style')
        ls_text = QtWidgets.QLineEdit(self.parseArtist(field='linestyle', default='--', return_type=str))
        ls_text.setToolTip('"-" (default), "--", "-.", ":"')
        lc_label = QtWidgets.QLabel('Line Color')
        lc_text = ColorDropDownCombobox(default=self.parseArtist(field='linecolor', default='k', return_type=str))
        # lc_text = QtWidgets.QLineEdit(self.parseArtist(field='linecolor', default='k', return_type=str))  # single letters or hex string
        lc_text.setToolTip('Single letter or hex value of the color')

        # make a dictionary of hte vlaues
        self.settingDict['x0'] = x0_text
        self.settingDict['y0'] = y0_text
        self.settingDict['x1'] = x1_text
        self.settingDict['y1'] = y1_text
        self.settingDict['linewidth'] = lw_text
        self.settingDict['linestyle'] = ls_text
        self.settingDict['linecolor'] = lc_text

        # Add the widget to the window
        widgetFrame.layout().addWidget(x0_label, 0, 0, 1, 1)
        widgetFrame.layout().addWidget(x0_text, 0, 1, 1, 1)
        widgetFrame.layout().addWidget(y0_label, 0, 2, 1, 1)
        widgetFrame.layout().addWidget(y0_text, 0, 3, 1, 1)
        widgetFrame.layout().addWidget(x1_label, 1, 0, 1, 1)
        widgetFrame.layout().addWidget(x1_text, 1, 1, 1, 1)
        widgetFrame.layout().addWidget(y1_label, 1, 2, 1, 1)
        widgetFrame.layout().addWidget(y1_text, 1, 3, 1, 1)
        widgetFrame.layout().addWidget(lw_label, 2, 2, 1, 1)
        widgetFrame.layout().addWidget(lw_text, 2, 3, 1, 1)
        widgetFrame.layout().addWidget(ls_label, 3, 0, 1, 1)
        widgetFrame.layout().addWidget(ls_text, 3, 1, 1, 1)
        widgetFrame.layout().addWidget(lc_label, 3, 2, 1, 1)
        widgetFrame.layout().addWidget(lc_text, 3, 3, 1, 1)

        return widgetFrame

    def ttlSettings(self):
        """return a dictionary of the TTL annotation artist"""
        widgetFrame = QtWidgets.QFrame()
        widgetFrame.setLayout(QtWidgets.QGridLayout())
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        widgetFrame.setSizePolicy(sizePolicy)
        widgetFrame.setObjectName(_fromUtf8("ttlSettingsWidgetFrame"))

        # Settings for TTL
        pulse2step_checkbox = QtWidgets.QCheckBox('Convert Pulse to Step')
        pulse2step_checkbox.setCheckState(2)
        pulse2step_checkbox.setToolTip("Draw a block of short pulses as a continuous step")

        realpulse_checkbox = QtWidgets.QCheckBox('Draw Real Pulse Width')
        realpulse_checkbox.setCheckState(2)
        realpulse_checkbox.setToolTip('If unchecked, pulses will be represented as a vertial line only')

        # Make a dictionary of the values
        self.settingDict['bool_pulse2step'] = pulse2step_checkbox
        self.settingDict['bool_realpulse'] = realpulse_checkbox

        # Add the widgets to the window
        widgetFrame.layout().addWidget(pulse2step_checkbox, 0, 0, 1, 1)
        widgetFrame.layout().addWidget(realpulse_checkbox, 1, 0, 1, 1)

        return widgetFrame


    def checkSettingUpdates(self):
        """Sanity check fields that cannot be empty input"""
        if self.type == 'box':
            keys = ['x0', 'y0', 'width', 'height', 'linewidth', 'linestyle', 'linecolor', 'fillcolor', 'fillalpha']
        elif self.type == 'line':
            keys = ['x0', 'y0', 'x1', 'y1', 'linewidth', 'linestyle', 'linecolor']
        elif self.type == 'circle':
            return True
        elif self.type == 'arrow':
            return True
        elif self.type == 'symbol':
            return True
        elif self.type == 'ttl':
            return True
        else:
            return True

        for k in keys:
            if self.artist[k] == '':
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("'{}' argument cannot be empty when drawing a '{}'".format(k, self.type))
                msg.exec_()
                return False
        return True

    def updateSettings(self, closeWidget=False):
        for k, v in self.settingDict.items():
            if isinstance(v, QtWidgets.QComboBox):
                val = v.currentText()
            elif isinstance(v, QtWidgets.QLineEdit):
                val = v.text()
            elif isinstance(v, QtWidgets.QCheckBox):
                val = True if v.checkState() > 0 else False
            elif isinstance(v, QtWidgets.QSpinBox):
                val = v.value()
            else:
                raise (TypeError('Unrecognized type of setting item'))

            self.artist[k] = val

        # sanity check
        state = self.checkSettingUpdates()
        if not state:
            return

        if closeWidget:
            self.accept()

    def parseArtist(self, field, default, return_type=None):
        if field in self.artist.keys():
            val = self.artist[field]
        else:
            val = default

        if return_type is not None:
            if return_type is bool:
                val = bool(val)*2 # for boolean check state
            else:
                val = return_type(val)

        return val


if __name__ == '__main__':
    #    iniPath = 'D:/Edward/Documents/Assignments/Scripts/Python/PySynapse/resources/config.ini'
    #    with fileinput.input(iniPath, inplace=True, backup='.bak') as f:
    #        for line in f:
    #            if line[0] == '#':
    #                print('#asdf.mat')
    #            else:
    #                print(line, end='')

    app = QtWidgets.QApplication(sys.argv)
    ex = AnnotationSetting()
    ex.show()
    if ex.exec_():
        print(ex.artist)
    # fff = app.exec_()
    # print(ex.artists)
    # sys.exit(fff)

