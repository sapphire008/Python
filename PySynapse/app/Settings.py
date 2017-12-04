# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 14:42:35 2016

Settings window

Interfrace for settings
Read and write ./resouces/config.ini text file

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

# <editor-fold desc="Global Settings">
# ------------ Read and write the settings file ----------------------------------
def readini(iniPath):
    """Read the saved config.ini for previous settings"""
    options = dict()
    with open(iniPath, 'r') as f: # read only
        # read line by line
        for line in f:
            if line[0] == '#': # skip comment line
                continue
            elif '#' in line: # get rid of inline comments
                line = line.split('#')[0]
            # Separate the line by '='
            key, val = [k.strip() for k in line.split('=')]
            # check if val is numeric string
            try:
                val = str2numeric(val)
            except:
                pass
            
            # check if val is boolean
            if isinstance(val, str):
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
            
            options[key] = val
    
    if not f.closed:
        f.close()
        
    return options
    
def writeini(iniPath, options):
    """Write the config.ini to save the current settings"""
    with fileinput.input(iniPath, inplace=True) as f:
        for line in f:
            if line[0] == '#' or line.strip() == '':
                print(line, end='')
                continue
            elif '#' in line: # temporarily store inline comments before writing
                params, comments = line.split('#')
                params = params.strip()
                comments = '#'+comments
            else:
                params, comments = line.strip(), ''
            
            # parse which key of the option dictionary in the current params
            for k, v in options.items():
                if k == params.split('=')[0].strip():
                    writeStr = '{} = {} {}'.format(k, str(v), comments).strip()
                    print(writeStr)
                    break

# ------------ Settings widget ---------------------------------------------------
class Settings(QtGui.QWidget):
    def __init__(self, parent=None, iniPath=None):
        super(Settings, self).__init__(parent)
        self.setWindowTitle("Settings")
        self.setWindowIcon(QtGui.QIcon('resources/icons/setting.png'))
        self.isclosed = True
        if iniPath == None:
            self.iniPath = os.path.join(__location__,'../resources/config.ini')
        else:
            self.iniPath = iniPath
        # Get the options: sets self.options
        self.options = readini(iniPath=self.iniPath)
        self.settingDict = {} # map between field name and objects that stores the setting
        # Set up the GUI
        self.setLayout(QtGui.QVBoxLayout())
        self.tabWidget = QtGui.QTabWidget()

        # Adding tabs
        self.tabWidget.addTab(self.exportTraceTabUI(),"Export")        
        self.tabWidget.addTab(self.viewTabUI(), 'View')

        # buttons for saving the settings and exiting the settings window
        OK_button = QtGui.QPushButton('OK')
        OK_button.setDefault(True)
        OK_button.clicked.connect(lambda: self.updateSettings(closeWidget=True))
        Apply_button = QtGui.QPushButton('Apply')
        Apply_button.clicked.connect(lambda: self.updateSettings(closeWidget=False))
        Cancel_button = QtGui.QPushButton('Cancel')
        Cancel_button.clicked.connect(self.close)
        self.buttonGroup = QtGui.QGroupBox()
        self.buttonGroup.setLayout(QtGui.QHBoxLayout())
        self.buttonGroup.layout().addWidget(OK_button, 0)
        self.buttonGroup.layout().addWidget(Apply_button, 0)
        self.buttonGroup.layout().addWidget(Cancel_button, 0)
        
        self.layout().addWidget(self.tabWidget)
        self.layout().addWidget(self.buttonGroup)

    #--------------- Set up the GUI ----------------------------------------------
    def exportTraceTabUI(self):
        widgetFrame = QtGui.QFrame()
        widgetFrame.setLayout(QtGui.QVBoxLayout())
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        widgetFrame.setSizePolicy(sizePolicy)
        widgetFrame.setObjectName(_fromUtf8("ExportTraceWidgetFrame"))

        # %% Size
        fig_size_W_label = QtGui.QLabel('Width (inches)')
        fig_size_W_text = QtGui.QLineEdit(str(self.options['figSizeW']))
        self.settingDict['figSizeW'] = fig_size_W_text
        fig_size_W_checkBox = QtGui.QCheckBox('Dynamically Adjust Width')
        fig_size_W_checkBox.setToolTip('Dynamically adjust the width of the figure when exporting multiple episodes')
        fig_size_W_checkBox.setCheckState(2 if self.options['figSizeWMulN'] else 0)
        self.settingDict['figSizeWMulN'] = fig_size_W_checkBox
        
        fig_size_H_label = QtGui.QLabel('Height (inches)')
        fig_size_H_text = QtGui.QLineEdit(str(self.options['figSizeH']))
        self.settingDict['figSizeH'] = fig_size_H_text
        fig_size_H_checkBox = QtGui.QCheckBox('Dynamically Adjust Height')
        fig_size_H_checkBox.setToolTip('Dynamically adjust the width of the figure when exporting multiple episodes')
        fig_size_H_checkBox.setCheckState(2 if self.options['figSizeHMulN'] else 0)
        self.settingDict['figSizeHMulN'] = fig_size_H_checkBox


        size_groupBox = QtGui.QGroupBox("Size")
        size_groupBox.setLayout(QtGui.QGridLayout())
        size_groupBox.layout().addWidget(fig_size_W_label, 0, 0, 1, 1)
        size_groupBox.layout().addWidget(fig_size_W_text, 0, 1, 1, 1)
        size_groupBox.layout().addWidget(fig_size_W_checkBox, 0, 2, 1, 2)
        size_groupBox.layout().addWidget(fig_size_H_label, 1, 0, 1, 1)
        size_groupBox.layout().addWidget(fig_size_H_text, 1, 1, 1, 1)
        size_groupBox.layout().addWidget(fig_size_H_checkBox, 1, 2, 1, 2)

        # %% Concatenated
        hSpace_label = QtGui.QLabel('Horizontal Space')
        hSpace_label.setToolTip('Only relevant when concatenating series of traces')
        hSpace_spinbox = QtGui.QSpinBox()
        hSpace_spinbox.setValue(self.options['hFixedSpace'])
        hSpace_spinbox.setSuffix('%')
        hSpace_spinbox.setRange(0,100)
        hSpace_spinbox.setFixedWidth(50)
        hSpace_comboBox = QtGui.QComboBox()
        hSpace_comboList = ['Fixed', 'Real Time']
        hSpace_comboBox.addItems(hSpace_comboList)
        hSpace_comboBox.setCurrentIndex(hSpace_comboList.index(self.options['hSpaceType']))
        hSpace_comboBox.currentIndexChanged.connect(lambda: self.toggleHFixedSpace(hSpace_comboBox, hSpace_spinbox, 'Real Time'))
        if hSpace_comboBox.currentText() == 'Real Time':
            hSpace_text.setEnabled(False)
        self.settingDict['hSpaceType'] = hSpace_comboBox
        # self.settingDict['hFixedSpace'] = hSpace_text
        self.settingDict['hFixedSpace'] = hSpace_spinbox
        
        concat_groupBox = QtGui.QGroupBox('Concatenated')
        concat_groupBox.setLayout(QtGui.QGridLayout())
        concat_groupBox.layout().addWidget(hSpace_label, 0, 0, 1,1)
        concat_groupBox.layout().addWidget(hSpace_comboBox, 0, 1, 1,1)
        concat_groupBox.layout().addWidget(hSpace_spinbox, 0, 3, 1,1)
        
        # %% Gridspec options
        gridSpec_label = QtGui.QLabel('Arrangement')
        gridSpec_label.setToolTip('Only relevant when exporting series of traces in a grid layout')
        gridSpec_comboBox = QtGui.QComboBox()
        gridSpec_comboList = ['Vertically', 'Horizontally', 'Channels x Episodes', 'Episodes x Channels']
        gridSpec_comboBox.addItems(gridSpec_comboList)
        gridSpec_comboBox.setCurrentIndex(gridSpec_comboList.index(self.options['gridSpec']))
        self.settingDict['gridSpec'] = gridSpec_comboBox
        
        scalebarAt_label = QtGui.QLabel('Scalebar Location')
        scalebarAt_label.setToolTip('Only relevant when exporting series of traces in a grid layout')
        scalebarAt_comboBox = QtGui.QComboBox()
        scalebarAt_comboList = ['All', 'First','Last','None']
        scalebarAt_comboBox.addItems(scalebarAt_comboList)
        scalebarAt_comboBox.setCurrentIndex(scalebarAt_comboList.index(self.options['scalebarAt']))
        self.settingDict['scalebarAt'] = scalebarAt_comboBox
        
        gridSpec_groupBox = QtGui.QGroupBox('Grid')
        gridSpec_groupBox.setLayout(QtGui.QGridLayout())
        gridSpec_groupBox.layout().addWidget(gridSpec_label, 0, 0, 1,1)
        gridSpec_groupBox.layout().addWidget(gridSpec_comboBox, 0, 1, 1, 1)
        gridSpec_groupBox.layout().addWidget(scalebarAt_label, 1, 0, 1,1)
        gridSpec_groupBox.layout().addWidget(scalebarAt_comboBox, 1,1, 1, 1)
        
        # %% output
        dpi_label = QtGui.QLabel('DPI')
        dpi_text = QtGui.QLineEdit(str(self.options['dpi']))
        self.settingDict['dpi'] = dpi_text

        linewidth_label = QtGui.QLabel('Linewidth')
        linewidth_text = QtGui.QLineEdit(str(self.options['linewidth']))
        self.settingDict['linewidth'] = linewidth_text
        
        fontName_label = QtGui.QLabel('Font Name')
        fontName_text = QtGui.QLineEdit(self.options['fontName'])
        self.settingDict['fontName'] = fontName_text
        fontSize_label = QtGui.QLabel('Font Size')
        fontSize_text = QtGui.QLineEdit(str(self.options['fontSize']))
        self.settingDict['fontSize'] = fontSize_text
        
        annotation_label = QtGui.QLabel('Annotation')
        annotation_comboBox = QtGui.QComboBox()
        ann_comboList = ['Label Only', 'Simple', 'Full', 'None']
        annotation_comboBox.addItems(ann_comboList)
        annotation_comboBox.setCurrentIndex(ann_comboList.index(self.options['annotation']))
        self.settingDict['annotation'] = annotation_comboBox

        monostim_checkbox = QtGui.QCheckBox('Force monochrome stim')
        monostim_checkbox.setToolTip('If checked, stimulus channel will not be color coded even when other channels are color coded')
        monostim_checkbox.setCheckState(2 if self.options['monoStim'] else 0)
        self.settingDict['monoStim'] = monostim_checkbox

        SRC_label = QtGui.QLabel('Stim=Current') # stim reflecting current
        SRC_checkbox = QtGui.QCheckBox()
        SRC_checkbox.setToolTip('If checked, stimulus will be shifted to baseline current level')
        SRC_checkbox.setCheckState(2 if self.options['stimReflectCurrent'] else 0)
        self.settingDict['stimReflectCurrent'] = SRC_checkbox

        showInitVal = QtGui.QCheckBox("Show Initial Value")
        showInitVal.setToolTip("Display the initial value at the beginning of the trace")
        showInitVal.setCheckState(2 if self.options['showInitVal'] else 0)
        self.settingDict['showInitVal'] = showInitVal
        
        saveDir_label = QtGui.QLabel('Path')
        saveDir_text = QtGui.QLineEdit(self.options['saveDir'])
        self.settingDict['saveDir'] = saveDir_text
        
        output_groupBox = QtGui.QGroupBox('Output')
        output_groupBox.setLayout(QtGui.QGridLayout())

        output_groupBox.layout().addWidget(annotation_label, 0, 0, 1, 1)
        output_groupBox.layout().addWidget(annotation_comboBox, 0, 1, 1, 1)
        output_groupBox.layout().addWidget(monostim_checkbox, 0, 2, 1, 2)
        output_groupBox.layout().addWidget(dpi_label, 1, 0, 1, 1)
        output_groupBox.layout().addWidget(dpi_text, 1, 1, 1, 1)
        output_groupBox.layout().addWidget(linewidth_label, 1, 2, 1, 1)
        output_groupBox.layout().addWidget(linewidth_text, 1, 3, 1, 1)
        output_groupBox.layout().addWidget(fontName_label, 2, 0, 1, 1)
        output_groupBox.layout().addWidget(fontName_text, 2, 1, 1, 1)
        output_groupBox.layout().addWidget(fontSize_label, 2, 2, 1, 1)
        output_groupBox.layout().addWidget(fontSize_text, 2, 3, 1, 1)
        output_groupBox.layout().addWidget(SRC_label, 3, 0, 1, 1)
        output_groupBox.layout().addWidget(SRC_checkbox, 3, 1, 1, 1)
        output_groupBox.layout().addWidget(showInitVal, 3, 2, 1, 1)
        output_groupBox.layout().addWidget(saveDir_label, 4, 0, 1, 1)
        output_groupBox.layout().addWidget(saveDir_text, 4, 1, 1, 3)
            
        # %% Organize widgets

        widgetFrame.layout().addWidget(size_groupBox)
        widgetFrame.layout().addWidget(concat_groupBox)
        widgetFrame.layout().addWidget(gridSpec_groupBox)
        widgetFrame.layout().addWidget(output_groupBox)
        widgetFrame.layout().addStretch(10)
        
        return widgetFrame

    def viewTabUI(self):
        widgetFrame = QtGui.QFrame()
        widgetFrame.setLayout(QtGui.QVBoxLayout())
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        widgetFrame.setSizePolicy(sizePolicy)
        widgetFrame.setObjectName(_fromUtf8("ViewWidgetFrame"))

        # Default View Range
        stream_label = QtGui.QLabel("Stream")
        min_label = QtGui.QLabel("Min")
        max_label = QtGui.QLabel("Max")
        time_label = QtGui.QLabel('Time')
        time_min_text = QtGui.QLineEdit(str(self.options['timeRangeMin']))
        time_max_text = QtGui.QLineEdit(str(self.options['timeRangeMax']))
        volt_label = QtGui.QLabel("Voltage")
        volt_min_text = QtGui.QLineEdit(str(self.options['voltRangeMin']))
        volt_max_text = QtGui.QLineEdit(str(self.options['voltRangeMax']))
        cur_label = QtGui.QLabel("Current")
        cur_min_text = QtGui.QLineEdit(str(self.options['curRangeMin']))
        cur_max_text = QtGui.QLineEdit(str(self.options['curRangeMax']))
        stim_label = QtGui.QLabel("Stimulus")
        stim_min_text = QtGui.QLineEdit(str(self.options['stimRangeMin']))
        stim_max_text = QtGui.QLineEdit(str(self.options['stimRangeMax']))

        # Put objects into setting dictionary
        self.settingDict['timeRangeMin'] = time_min_text
        self.settingDict['timeRangeMax'] = time_max_text
        self.settingDict['voltRangeMin'] = volt_min_text
        self.settingDict['voltRangeMax'] = volt_max_text
        self.settingDict['curRangeMin'] = cur_min_text
        self.settingDict['curRangeMax'] = cur_max_text
        self.settingDict['stimRangeMin'] = stim_min_text
        self.settingDict['stimRangeMax'] = stim_max_text

        # Add to the groupbox
        view_groupBox = QtGui.QGroupBox('Default Range')
        view_groupBox.setLayout(QtGui.QGridLayout())
        view_groupBox.layout().addWidget(stream_label, 0, 0, 1, 1)
        view_groupBox.layout().addWidget(min_label, 0, 1, 1, 1)
        view_groupBox.layout().addWidget(max_label, 0, 2, 1, 1)
        view_groupBox.layout().addWidget(time_label, 1, 0, 1, 1)
        view_groupBox.layout().addWidget(time_min_text, 1, 1, 1, 1)
        view_groupBox.layout().addWidget(time_max_text, 1, 2, 1, 1)
        view_groupBox.layout().addWidget(volt_label, 2, 0, 1, 1)
        view_groupBox.layout().addWidget(volt_min_text, 2, 1, 1, 1)
        view_groupBox.layout().addWidget(volt_max_text, 2, 2, 1, 1)
        view_groupBox.layout().addWidget(cur_label, 3, 0, 1, 1)
        view_groupBox.layout().addWidget(cur_min_text, 3, 1, 1, 1)
        view_groupBox.layout().addWidget(cur_max_text, 3, 2, 1, 1)
        view_groupBox.layout().addWidget(stim_label, 4, 0, 1, 1)
        view_groupBox.layout().addWidget(stim_min_text, 4, 1, 1, 1)
        view_groupBox.layout().addWidget(stim_max_text, 4, 2, 1, 1)

        # Oragnize the widget
        widgetFrame.layout().addWidget(view_groupBox)
        widgetFrame.layout().addStretch(10)

        return widgetFrame

    def updateSettings(self, closeWidget=False):
        for k, v in self.settingDict.items():
            if isinstance(v, QtGui.QComboBox):
                val = v.currentText()
            elif isinstance(v, QtGui.QLineEdit):
                val = v.text()
            elif isinstance(v, QtGui.QCheckBox):
                val = True if v.checkState()>0 else False
            elif isinstance(v, QtGui.QSpinBox):
                val = v.value()
            else:
                raise(TypeError('Unrecognized type of setting item'))
            
            self.options[k] = val
            
        # save all the parameters
        writeini(iniPath=self.iniPath, options=self.options)
            
        if closeWidget:
            self.close()
            
    def toggleHFixedSpace(self, hSpace_comboBox, hSpace_spinbox, forbidden_text):
        if hSpace_comboBox.currentText() == forbidden_text:
            hSpace_spinbox.setEnabled(False)
        else:
            hSpace_spinbox.setEnabled(True)
        
    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        self.isclosed = True
        
# </editor-fold>


if __name__ == '__main__':
#    iniPath = 'D:/Edward/Documents/Assignments/Scripts/Python/PySynapse/resources/config.ini'
#    with fileinput.input(iniPath, inplace=True, backup='.bak') as f:
#        for line in f:
#            if line[0] == '#':
#                print('#asdf.mat')
#            else:
#                print(line, end='')
        
   app = QtGui.QApplication(sys.argv)
   ex = Settings()
   ex.show()
   sys.exit(app.exec_())