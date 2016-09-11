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
        
        self.tabWidget.addTab(self.exportTraceTabUI(),"Export")        
        
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
        widgetFrame.setLayout(QtGui.QGridLayout())
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        widgetFrame.setSizePolicy(sizePolicy)
        widgetFrame.setObjectName(_fromUtf8("ExportTraceWidgetFrame"))
                
        fig_size_W_label = QtGui.QLabel('Width (inches)')
        fig_size_W_text = QtGui.QLineEdit(str(self.options['figSizeW']))
        self.settingDict['figSizeW'] = fig_size_W_text
        fig_size_W_checkBox = QtGui.QCheckBox(u'\u00D7 # channels / streams')
        fig_size_W_checkBox.setCheckState(2 if self.options['figSizeWMulN'] else 0)
        self.settingDict['figSizeWMulN'] = fig_size_W_checkBox
        
        fig_size_H_label = QtGui.QLabel('Height (inches)')
        fig_size_H_text = QtGui.QLineEdit(str(self.options['figSizeH']))
        self.settingDict['figSizeH'] = fig_size_H_text
        fig_size_H_checkBox = QtGui.QCheckBox(u'\u00D7 # channels / streams')
        fig_size_H_checkBox.setCheckState(2 if self.options['figSizeHMulN'] else 0)
        self.settingDict['figSizeHMulN'] = fig_size_H_checkBox
        
        dpi_label = QtGui.QLabel('DPI')
        dpi_text = QtGui.QLineEdit(str(self.options['dpi']))
        self.settingDict['dpi'] = dpi_text
        
        fontName_label = QtGui.QLabel('Font Name')
        fontName_text = QtGui.QLineEdit(self.options['fontName'])
        self.settingDict['fontName'] = fontName_text
        fontSize_label = QtGui.QLabel('Font Size')
        fontSize_text = QtGui.QLineEdit(str(self.options['fontSize']))
        self.settingDict['fontSize'] = fontSize_text
        
        annotation_label = QtGui.QLabel('Annotation')
        annotation_comboBox = QtGui.QComboBox()
        comboList = ['Simple', 'Full', 'None']
        annotation_comboBox.addItems(comboList)
        annotation_comboBox.setCurrentIndex(comboList.index(self.options['annotation']))
        self.settingDict['annotation'] = annotation_comboBox
        
        saveDir_label = QtGui.QLabel('Path')
        saveDir_text = QtGui.QLineEdit(self.options['saveDir'])
        self.settingDict['saveDir'] = saveDir_text
            
        # Organize widgets
        widgetFrame.layout().addWidget(fig_size_W_label, 0, 0, 1, 1)
        widgetFrame.layout().addWidget(fig_size_W_text, 0, 1, 1, 1)
        widgetFrame.layout().addWidget(fig_size_W_checkBox, 0, 2, 1, 2)
        widgetFrame.layout().addWidget(fig_size_H_label, 1, 0, 1, 1)
        widgetFrame.layout().addWidget(fig_size_H_text, 1, 1, 1, 1)
        widgetFrame.layout().addWidget(fig_size_H_checkBox, 1, 2, 1, 2)
        widgetFrame.layout().addWidget(dpi_label, 2, 0, 1, 1)
        widgetFrame.layout().addWidget(dpi_text, 2, 1, 1, 1)
        widgetFrame.layout().addWidget(annotation_label, 3, 0, 1, 1)
        widgetFrame.layout().addWidget(annotation_comboBox, 3, 1, 1, 1)
        widgetFrame.layout().addWidget(fontName_label, 4, 0, 1, 1)
        widgetFrame.layout().addWidget(fontName_text, 4, 1, 1, 1)
        widgetFrame.layout().addWidget(fontSize_label, 4, 2, 1, 1)
        widgetFrame.layout().addWidget(fontSize_text, 4, 3, 1, 1)
        widgetFrame.layout().addWidget(saveDir_label, 5, 0, 1, 1)
        widgetFrame.layout().addWidget(saveDir_text, 5, 1, 1, 3)
        
        return widgetFrame
        
    def updateSettings(self, closeWidget=False):
        for k, v in self.settingDict.items():
            if isinstance(v, QtGui.QComboBox):
                val = v.currentText()
            elif isinstance(v, QtGui.QLineEdit):
                val = v.text()
            elif isinstance(v, QtGui.QCheckBox):
                val = True if v.checkState()>0 else False
            else:
                raise(TypeError('Unrecognized type of setting item'))
            
            self.options[k] = val
            
        # save all the parameters
        writeini(iniPath=self.iniPath, options=self.options)
            
        if closeWidget:
            self.close()
        
    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        self.isclosed = True
        


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