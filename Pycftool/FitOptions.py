"""
A variety of interfaces for fit options

"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../generic/")))

from MATLAB import *

from PyQt5 import QtCore, QtGui, QtWidgets
from QCodeEdit import QCodeEdit
from ElideQLabel import ElideQLabel


import numpy as np


def str2numericHandleError(x):
    """Wrapper fir str2numeric. Return original string instead of error """
    try:
        return str2numeric(x)
    except:
        return x

class FitOptions(QtWidgets.QWidget):
    def __init__(self, parent=None, friend=None, method='2D: curve_fit', coefficients=['a', 'b', 'c']):
        super(FitOptions, self).__init__(parent)
        self.setWindowTitle("Fit Options")
        self.parent = parent
        self.friend = friend
        self.isclosed = True  # start off being closed
        self.method = method
        self.params = {'method': method}
        self.coefficients = coefficients

        # Set up GUI
        self.setLayout(QtWidgets.QVBoxLayout())

        # buttons for saving the settings and exiting the options window
        OK_button = QtWidgets.QPushButton('OK')
        OK_button.setDefault(True)
        OK_button.clicked.connect(lambda: self.updateSettings(closeWidget=True))
        Apply_button = QtWidgets.QPushButton('Apply')
        Apply_button.clicked.connect(lambda: self.updateSettings(closeWidget=False))
        Cancel_button = QtWidgets.QPushButton('Cancel')
        Cancel_button.clicked.connect(self.close)
        self.buttonGroup = QtWidgets.QGroupBox()
        self.buttonGroup.setLayout(QtWidgets.QHBoxLayout())
        self.buttonGroup.layout().addWidget(OK_button, 0)
        self.buttonGroup.layout().addWidget(Apply_button, 0)
        self.buttonGroup.layout().addWidget(Cancel_button, 0)

        # Populate the fit options regions with appropriate field
        self.initializeFitOptionsWidget()

        self.layout().addWidget(self.widgets)
        self.layout().addWidget(self.buttonGroup)

    def setMethod(self, method):
        if self.method == method:
            return
        self.method = method
        # remove the current widget
        self.layout().removeWidget(self.widgets)
        self.widgets.deleteLater()
        # re-initialize self.widgets
        self.initializeFitOptionsWidget()
        # Replace the current widget
        self.layout().insertWidget(0, self.widgets)
        # print('reset method to --> {}'.format(method))

    def setInitializationParameters(self, coefficients=['a','b','c','d']):
        self.coefficients = coefficients
        coef_table = QtWidgets.QTableWidget(0, 4)
        coef_table.verticalHeader().setVisible(False)
        coef_table.setHorizontalHeaderLabels(['Coefficients', 'StartingPoint', 'Lower', 'Upper'])
        coef_table.itemChanged.connect(lambda: self.refitCurve())
        coef_table.blockSignals(True)  # blocking any signal while setting data

        coef_flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable

        for n, pp in enumerate(coefficients):
            coef_table.insertRow(n)  # add a row
            # Coefficient label
            coef_label = QtWidgets.QTableWidgetItem(pp)
            # Initial Value
            initVal_item = QtWidgets.QTableWidgetItem("{:.5f}".format(np.random.rand()))
            initVal_item.setFlags(coef_flags)
            lower_item = QtWidgets.QTableWidgetItem(str(-np.inf))
            lower_item.setFlags(coef_flags)
            upper_item = QtWidgets.QTableWidgetItem(str(np.inf))
            upper_item.setFlags(coef_flags)

            coef_table.setItem(n, 0, coef_label)
            coef_table.setItem(n, 1, initVal_item)
            coef_table.setItem(n, 2, lower_item)
            coef_table.setItem(n, 3, upper_item)

        coef_table.blockSignals(False)  # release signal block

        # Resize table cells to fit contents
        coef_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        coef_table.resizeColumnsToContents()
        coef_table.resizeRowsToContents()

        # Resetting parameter associated with coef table, if already initialized
        if hasattr(self, 'widgets') and hasattr(self.widgets, 'params'):
            # Delete the table
            table_row = self.widgets.layout().rowCount()-1
            self.widgets = self.friend.removeFromWidget(self.widgets, row=table_row)
            # Add back the new table
            self.widgets.layout().addWidget(coef_table, table_row, 0, 1, 2)
            # Update params
            self.widgets.params['coefficients'] = coef_table

        return coef_table

    def initializeFitOptionsWidget(self):
        if self.method == '2D: curve_fit':  # General call to curve_fit
            self.widgets = self.curveFitOptionWidgets()
        elif self.method == '2D: gmm':  # Gaussian mixture model
            self.widgets = QtWidgets.QGroupBox()
        elif self.method == '2D: fourier':  # Mixture of fourier
            self.widgets = QtWidgets.QGroupBox()
        elif self.method == '3D: gmm':  # 3D Gaussian mixture model
            self.widgets = QtWidgets.QGroupBox()
        else:
            self.widgets = QtWidgets.QGroupBox()

    def curveFitOptionWidgets(self):
        widgets = QtWidgets.QGroupBox()
        widgets.setLayout(QtWidgets.QGridLayout())

        # <editor-fold desc="Options">
        method_label = QtWidgets.QLabel("Method:")
        method_detail_label = QtWidgets.QLabel("NonlinearLeastSquares")
        algorithm_label = QtWidgets.QLabel("Algorithm:")
        algorithm_comboBox = QtWidgets.QComboBox()
        algorithm_comboBox.addItems(["Trust-Region Reflective", "Levenberg-Marquardt", "Dog-Box"])
        algorithm_comboBox.setItemData(0, "Trust Region Reflective (TRF) algorithm,\nparticularly suitable for large sparse problems with bounds.\nGenerally robust method.", QtCore.Qt.ToolTipRole)
        algorithm_comboBox.setItemData(1, "Levenberg-Marquardt (LM) algorithm as implemented in MINPACK.\nDoesn’t handle bounds and sparse Jacobians.\nUsually the most efficient method for small unconstrained problems.", QtCore.Qt.ToolTipRole)
        algorithm_comboBox.setItemData(2, "Dogleg algorithm (dogbox) with rectangular trust regions,\ntypical use case is small problems with bounds.\nNot recommended for problems with rank-deficient Jacobian.", QtCore.Qt.ToolTipRole)
        algorithm_comboBox.setToolTip("Expand to see details")
        algorithm_comboBox.currentIndexChanged.connect(lambda: self.refitCurve())
        maxfev_label = QtWidgets.QLabel("maxfeval:")
        maxfev_LineEdit = QtWidgets.QLineEdit("None")
        maxfev_LineEdit.setToolTip("The maximum number of calls to the function.\nSet to 0 for automatic calculation.")
        maxfev_LineEdit.returnPressed.connect(lambda: self.refitCurve())
        loss_label = QtWidgets.QLabel("loss:")
        loss_comboBox = QtWidgets.QComboBox()
        loss_comboBox.addItems(['linear', 'soft_l1', 'huber', 'cauchy', 'arctan'])
        loss_comboBox.setToolTip("Determines the loss function")
        loss_comboBox.setItemData(0, "rho(z) = z.\nGives a standard least-squares problem.", QtCore.Qt.ToolTipRole)
        loss_comboBox.setItemData(1, "rho(z) = 2 * ((1 + z)**0.5 - 1).\nThe smooth approximation of l1 (absolute value) loss.\nUsually a good choice for robust least squares.", QtCore.Qt.ToolTipRole)
        loss_comboBox.setItemData(2, "rho(z) = z if z <= 1 else 2*z**0.5 - 1.\nWorks similarly to ‘soft_l1’.", QtCore.Qt.ToolTipRole)
        loss_comboBox.setItemData(3, "rho(z) = ln(1 + z).\nSeverely weakens outliers influence,\nbut may cause difficulties in optimization process.", QtCore.Qt.ToolTipRole)
        loss_comboBox.setItemData(4, "rho(z) = arctan(z).\nLimits a maximum loss on a single residual,\nhas properties similar to ‘cauchy’.", QtCore.Qt.ToolTipRole)
        loss_comboBox.currentIndexChanged.connect(lambda: self.refitCurve())
        ftol_label = QtWidgets.QLabel('ftol:')
        ftol_lineEdit = QtWidgets.QLineEdit('1.0e-08')
        ftol_lineEdit.setToolTip("Relative error desired in the sum of squares.")
        ftol_lineEdit.returnPressed.connect(lambda: self.refitCurve())
        xtol_label = QtWidgets.QLabel("xtol:")
        xtol_LineEdit = QtWidgets.QLineEdit('1.0e-08')
        xtol_LineEdit.setToolTip("Relative error desired in the approximate solution.")
        xtol_LineEdit.returnPressed.connect(lambda: self.refitCurve())
        gtol_label = QtWidgets.QLabel("gtol:")
        gtol_LineEdit = QtWidgets.QLineEdit("1.0e-08")
        gtol_LineEdit.setToolTip("Orthogonality desired between the function vector\nand the columns of the Jacobian.")
        gtol_LineEdit.returnPressed.connect(lambda: self.refitCurve())
        # </editor-fold>

        # Coefficients table
        coef_table = self.setInitializationParameters(coefficients=self.coefficients)

        # <editor-fold desc="Adding widgets">
        widgets.layout().addWidget(method_label, 0, 0, 1, 1)
        widgets.layout().addWidget(method_detail_label, 0, 1, 1, 1)
        widgets.layout().addWidget(algorithm_label, 1, 0, 1, 1)
        widgets.layout().addWidget(algorithm_comboBox, 1, 1, 1, 1)
        widgets.layout().addWidget(loss_label, 2, 0, 1, 1)
        widgets.layout().addWidget(loss_comboBox, 2, 1, 1, 1)
        widgets.layout().addWidget(maxfev_label, 3, 0, 1, 1)
        widgets.layout().addWidget(maxfev_LineEdit, 3, 1, 1, 1)
        widgets.layout().addWidget(ftol_label, 4, 0, 1, 1)
        widgets.layout().addWidget(ftol_lineEdit, 4, 1, 1, 1)
        widgets.layout().addWidget(xtol_label, 5, 0, 1, 1)
        widgets.layout().addWidget(xtol_LineEdit, 5, 1, 1, 1)
        widgets.layout().addWidget(gtol_label, 6, 0, 1, 1)
        widgets.layout().addWidget(gtol_LineEdit, 6, 1, 1, 1)
        widgets.layout().addWidget(coef_table, 7, 0, 1, 2)

        widgets.params = {'algorithm': algorithm_comboBox,
                          'loss': loss_comboBox,
                          'maxfev': maxfev_LineEdit,
                          'ftol': ftol_lineEdit,
                          'xtol': xtol_LineEdit,
                          'gtol': gtol_LineEdit,
                          'coefficients': coef_table}

        # </editor-fold>

        return widgets

    def gmmFitOptionWidgets(self):
        widgets = QtWidgets.QGroupBox()
        widgets.setLayout(QtWidgets.QGridLayout())

    def getParams(self):
        """From self.widget"""
        for key, widget in self.widgets.params.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                val = str2numericHandleError(widget.text())
            elif isinstance(widget, QtWidgets.QComboBox):
                val = str2numericHandleError(widget.currentText())
            elif isinstance(widget, QtWidgets.QLabel):
                val = str2numericHandleError(widget.text())
            elif isinstance(widget, QtWidgets.QCheckBox):
                val = True if widget.checkState() > 0 else False
            elif isinstance(widget, QtWidgets.QSpinBox):
                val = widget.value()

            elif isinstance(widget, QtWidgets.QTableWidget):
                val = {}
                numcols = widget.columnCount()
                for r in range(widget.rowCount()):
                    current_coef = widget.item(r, 0).text()
                    val[current_coef] = [str2numericHandleError(widget.item(r, c).text()) for c in range(1, numcols)]
            else:
                raise (TypeError('Unrecognized type of setting item'))

            self.params[key] = val

    def refitCurve(self):
        print('refitting...')
        if self.friend is None or not self.friend.autofit:
            return # don't do anything if not auto fitting
#
        self.friend.curveFit()

    def updateSettings(self, closeWidget=False):
        self.getParams() # Get the parameters
        if closeWidget:
            self.isclosed = True
            self.close()

    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        self.isclosed = True

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = FitOptions()
    ex.show()
    sys.exit(app.exec_())
