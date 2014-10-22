# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_MainWindow.ui'
#
# Created: Tue Aug 19 14:14:21 2014
#      by: Philipp Metzner, TUM E17 
#          using PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
from ptypySubclasses import PtypyTreeView

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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(911, 608)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.splitter.setChildrenCollapsible(False)
        self.verticalLayoutWidget = QtGui.QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        # !!! CRUCIAL LINE !!! 
        #self.treeView = QtGui.QTreeView(self.verticalLayoutWidget)
        self.treeView = PtypyTreeView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeView.sizePolicy().hasHeightForWidth())
        self.treeView.setSizePolicy(sizePolicy)
        self.treeView.setMinimumSize(QtCore.QSize(300, 300))
        self.treeView.setObjectName(_fromUtf8("treeView"))
        self.verticalLayout_2.addWidget(self.treeView)
        
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.Add_scan_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.Add_scan_button.setObjectName(_fromUtf8("Add_scan_button"))
        self.horizontalLayout_3.addWidget(self.Add_scan_button)
        self.Edit_scan_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.Edit_scan_button.setObjectName(_fromUtf8("Edit_scan_button"))
        self.horizontalLayout_3.addWidget(self.Edit_scan_button)
        self.Remove_scan_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.Remove_scan_button.setObjectName(_fromUtf8("Remove_scan_button"))
        self.horizontalLayout_3.addWidget(self.Remove_scan_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.New_engine_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.New_engine_button.setObjectName(_fromUtf8("New_engine_button"))
        self.horizontalLayout_4.addWidget(self.New_engine_button)
        self.Duplicate_engine_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.Duplicate_engine_button.setObjectName(_fromUtf8("Duplicate_engine_button"))
        self.horizontalLayout_4.addWidget(self.Duplicate_engine_button)
        self.Remove_engine_button = QtGui.QPushButton(self.verticalLayoutWidget)
        self.Remove_engine_button.setObjectName(_fromUtf8("Remove_engine_button"))
        self.horizontalLayout_4.addWidget(self.Remove_engine_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.frame = QtGui.QFrame(self.splitter)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label = QtGui.QLabel(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(300, 300))
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 911, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        
        self.actionLoad = QtGui.QAction(MainWindow)
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.actionSave = QtGui.QAction(MainWindow)
        self.actionSave.setObjectName(_fromUtf8("actionSave"))
        self.actionSave_As = QtGui.QAction(MainWindow)
        self.actionSave_As.setObjectName(_fromUtf8("actionSave_As"))
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName(_fromUtf8("actionAbout"))
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.Add_scan_button.setText(_translate("MainWindow", "&Add scan", None))
        self.Edit_scan_button.setText(_translate("MainWindow", "&Edit scan", None))
        self.Remove_scan_button.setText(_translate("MainWindow", "&Remove scan", None))
        self.New_engine_button.setText(_translate("MainWindow", "&New engine", None))
        self.Duplicate_engine_button.setText(_translate("MainWindow", "&Duplicate engine", None))
        self.Remove_engine_button.setText(_translate("MainWindow", "Re&move engine", None))
        self.label.setText(_translate("MainWindow", "Welcome to ptypy v0.0.1", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuHelp.setTitle(_translate("MainWindow", "Help", None))
        self.actionLoad.setText(_translate("MainWindow", "Load...", None))
        self.actionSave.setText(_translate("MainWindow", "Save", None))
        self.actionSave_As.setText(_translate("MainWindow", "Save As...", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionAbout.setText(_translate("MainWindow", "About", None))

