#!/usr/bin/python
# -*- coding: utf-8 -*-

# ptypy version 0.0.1
# 
# author: Philipp Metzner



from PyQt4 import QtXml, QtGui, QtCore
from Ui_MainWindow import Ui_MainWindow
from ptypySubclasses import *
import datetime, os, csv

CSV_INPUTPATH = 'NEW-ptypy-template.csv'
XML_INPUTPATH = 'ptypy_template.xml'
ENGINES_INPUTPATH = 'engines_template.xml'
PYTHON_OUTPUTPATH = 'final_python_output.py'

#PtypyDict = PtypyDictionaries(CSV_INPUTPATH)

class PtypyMainWindow(QtGui.QMainWindow, Ui_MainWindow):
    
    def __init__(self, app, parent=None):
        super(PtypyMainWindow, self).__init__()
        self.setupUi(self)
                
        # SET UP MODEL, TREEVIEW AND WINDOW TITLE (treeView is defined in Ui_MainWindow.py)
        #self.model = PtypyModel(inputPath=XML_INPUTPATH)
        self.model = PtypyModel(setupAtInit=False)
        self.loadCSVfile()
        self.treeView.setModel(self.model)
        self.treeView.setLayout()
        #self.treeView.hideFileAndSourcetypeRows()
        self.treeView.setCurrentIndex(self.model.index(0, 0))
        #self.enginesModel = PtypyModel(inputPath=ENGINES_INPUTPATH)
        self.setWindowTitle('%s - ptypy v0.0.1' % XML_INPUTPATH)
        
        # ATTRIBUTES
        self.contentModified = False
        # for reference when adding/editing "scan"s and "engine"s 
        self.scansItem = self.model.findItems('scans', QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0] 
        self.enginesItem = self.model.findItems('engines', QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
        
        # CONNECTIONS
        self.actionLoad.triggered.connect(self.loadFile)
        self.actionSave.triggered.connect(self.saveFile)
        self.actionSave_As.triggered.connect(self.saveFileAs)
        self.actionExit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.aboutDialog)
        
        self.Add_scan_button.clicked.connect(self.addScan)
        self.Edit_scan_button.clicked.connect(self.editScan)
        self.Remove_scan_button.clicked.connect(self.removeScanOrEngine)
        self.New_engine_button.clicked.connect(self.newEngine)
        self.Duplicate_engine_button.clicked.connect(self.duplicateEngine)
        self.Remove_engine_button.clicked.connect(self.removeScanOrEngine)
        
        self.treeView.selectionModel().selectionChanged.connect(self.showDocs)
        self.model.itemChanged.connect(self.treeView.modifyValue)
        #self.model.itemChanged.connect(self.updateScanParameters)
        
        # KEYBOARD SHORTCUTS
        self.actionLoad.setShortcut('Ctrl+L')
        self.actionSave.setShortcut('Ctrl+S')
        self.actionSave_As.setShortcut('Ctrl+Shift+S')
        self.actionExit.setShortcut('Ctrl+Q')
        
        
    #####
    # SLOTS FOR FILE, DOCUMENTATION DISPLAYING AND WINDOW CLOSING     
    #####
    def saveFileAs(self):
        """
        Opens 'Save File As' dialog so that user can give a path for the output xml file.
        """
        if self.model.hasChildren():
            outputPath = unicode(QtGui.QFileDialog.getSaveFileName(self, 'Save file',
            QtCore.QDir.currentPath(), 'xml file (*.xml)'))
            if len(outputPath) == 0:
                return False # Cancel pressed
            # add xml suffix if the user did not specify
            if outputPath[-4:] != '.xml':
                outputPath += '.xml'
            self.model.setOutputPath(outputPath)
            self.contentModified = True
            return self.saveFile()
        else:
            QtGui.QMessageBox.warning(self, 
            'Saving empty tree not possible', 
            'Please load a file first. You cannot save an empty tree.')
    
    
    def saveFile(self):
        """
        Usual 'Save File' procedure.
        When user saves template file for the very first time, saveFileAs() is called
        """
        if len(self.model.getOutputPath()) == 0:
            self.saveFileAs()
        else:
            if self.contentModified or self.treeView.contentModified: 
                self.model.writeModelToXML()
                self.convertXMLtoPython(self.model.getOutputPath())
                self.contentModified = self.treeView.contentModified = False
                return True
            else:
                QtGui.QMessageBox.information(self, 
                'File already saved', 
                'This version has already been saved. You did not apply any changes since the last saving time.\n' + 
                'If you intend to save the version with a different name, click \"Save File As\" or type Ctrl+Shift+S.')
    
    
    @QtCore.pyqtSlot()
    def aboutDialog(self):
        QtGui.QMessageBox.about(self, 
        'About ptypy v0.0.1',
        'GUI design and development by Philipp Metzner \nBased on the ptypy reconstruction software by ' + 
        'Bjoern Enders and Pierre Thibault \nTUM E17 2014')


    @QtCore.pyqtSlot()
    def loadFile(self):
        """
        Usual 'Load File' procedure. 
        Model, treeView, scansItem and windowTitle are set up with the fresh inputpath.
        """
        loadFileName = QtGui.QFileDialog.getOpenFileName(self, 'Load file',
            QtCore.QDir.currentPath(), 'xml file (*.xml)')
        if not loadFileName.isNull():
            self.model.clear()
            self.model.setHorizontalHeaderLabels(['Name', 'Value'])
            loadFileName = unicode(loadFileName)
            self.model.setInputPath(loadFileName)
            self.model.setOutputPath(loadFileName)
            self.model.parseXMLtoModel()
            self.treeView.setLayout()
            self.treeView.hideFileAndSourcetypeRows()
            self.setWindowTitle('%s - ptypy v0.0.1' % loadFileName)
            # remove the MatchRecursive flag
            self.scansItem = self.model.findItems('scans', QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
            self.enginesItem = self.model.findItems('engines', QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
    
    
    def closeEvent(self, event):
        """
        Reimplementation, called when X in corner, 'Exit' or 'Ctrl+Q' is pressed.
        Asks the user to save the file if changed since last saving.
        """ 
        if self.contentModified or self.treeView.contentModified:
            answer = QtGui.QMessageBox.question(self, 
            'Save file before closing?',
            'The file has been changed. Would you like to save it before closing?',
            QtGui.QMessageBox.Yes, QtGui.QMessageBox.No, QtGui.QMessageBox.Cancel)
            if answer == QtGui.QMessageBox.Yes:
                if self.saveFile():
                    event.accept()
                else:
                    # 'Cancel' is pressed in SaveFileAs dialog, prevent closing of MainWindow
                    event.ignore()
            elif answer == QtGui.QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept() 
    
    
    def showDocs(self, selection):
        """
        Show the documentation corresponding to the name of the clicked Item on the right.
        Sidenote: was showDocs(self, index) when connected to treeView.clicked signal before
        """
        index = selection.indexes()[0]
        #name = self.model.itemFromIndex(index).name
        line = self.model.itemFromIndex(index).docID
        try:
            text = "<b>Parameter</b>:<br />" + unicode(CSVdoc.name[line]) + "<br /><br />" + \
                   "<b>Type</b>:<br />" + unicode(CSVdoc.type[line]) + "<br /><br />" + \
                   "<b>Description</b>:<br />" + unicode(CSVdoc.shortdoc[line]) + "<br /><br />" + \
                   "<b>Information</b>:<br />"  + unicode(CSVdoc.longdoc[line]) + "<br /><br />" + \
                   "<b> Documentation ID:</b>" + unicode(str(line)) 
        except KeyError, err:
            text = "No documentation entry defined yet for %s" % str(err)[1:-1]
        self.label.setText(text)


    #####
    # SLOTS FOR SCAN AND ENGINE HANDLING
    #####
    def addScan(self):
        """
        Adds a 'scan' child with user-defined parameters that differ from the 'model' parameters.
        Values in the EditWindow() popup are compared with the 'model' ones and - if different - 
        appended to the corresponding 'scan' child.
        """
        model = self.model
        scansItem = self.scansItem
        assert scansItem.name == 'scans'
        
        scanName = "scan%03i" % (scansItem.counter+1)
        scanTemplatePath = '.temp-scan-template.xml'
        scanItem = Item(name=scanName,docID=CSVdoc.name.index('model'), specialType=1)

        templateItem = self.model.findItems('model', QtCore.Qt.MatchExactly)[0]
        self.writeEditTemplateToXML(scanTemplatePath, templateItem)
        
        scanWindow = EditWindow(parent=scanItem)
        scanWindow.model.setInputPath(scanTemplatePath)
        scanWindow.model.parseXMLtoModel()
        scanWindow.treeView.setLayout()
        
        # True if OK of EditWindow pressed
        if scanWindow.exec_(): 
            # append 'scan00X' row (== insert at position 'end')
            end = scansItem.rowCount()
            scansItem.insertRow(end, [scanItem, Item(name=scanName, isDummy=True)])
            scansItem.counter += 1
            self.compareScanParameters(templateItem, scanItem.editModel.item(0, 0), scansItem)
            self.contentModified = True
        os.remove(scanTemplatePath)

    
    def writeEditTemplateToXML(self, outputPath, templateItem):
        """
        Writes the Item of the current global 'model' parameters.
        (that are to be edited in EditWindow) to xml file 'outputPath' (attribute of ScanItem).
        """
        xmlWriter = QtCore.QXmlStreamWriter()
        # do indents and paragraphs in xml output
        xmlWriter.setAutoFormatting(True) 
        xmlFile = QtCore.QFile(outputPath)
        
        if (xmlFile.open(QtCore.QIODevice.WriteOnly) == False):
            QtGui.QMessageBox.warning(0, "Error!", "Error opening file")
        else:
            xmlWriter.setDevice(xmlFile)
            xmlWriter.writeStartDocument()
            xmlWriter.writeStartElement('root')
            templateItem.writeChildrenToXML(Item(isDummy=True), xmlWriter)
            xmlWriter.writeEndElement()
            xmlWriter.writeEndDocument()
            
    
    def compareScanParameters(self, templateItem, scanItem, appendItem):
        """
        Compare values of 'scan' parameters (from EditWindow) and 'model' parameters (from MainWindow).
        appendItem requires to be passed as reference for row insertion.
        """
        def appendChangedParameters(tn, tv, sn, sv, appender):
            """
            recursive function to compare 'model' and 'scan' parameters
            somehow only the nameItems in the left column "carry" the children
            hence are passed: templateItemName, -Value, scanItemName, -Value, rowToAppend
            """
            if not sv.isDummy: # deepest branch reached
                assert tv.name == sv.name, "%s - %s" % (tv.name, sv.name)
                if tv.value != sv.value :
                    name = Item(name=tv.name)
                    value = Item(name=tv.name, value=sv.value)
                    appender.insertRow(appender.rowCount(), [name, value])
                    self.changed = True
            else: # still a parent Item
                snOld = sn
                tnOld = tn
                appender.insertRow(appender.rowCount(), 
                                   [Item(name=sn.name), Item(name=sn.name, isDummy=True)])
                self.changed = False
                # loop over all child parameters
                for n in range(sn.rowCount()):
                    sn, sv = snOld.child(n), snOld.child(n, 1)
                    tn, tv = tnOld.child(n), tnOld.child(n, 1)
                    appendChangedParameters(tn, tv, sn, sv, appender.child(appender.rowCount()-1))
                if not self.changed:
                    # remove parent row again if no child has been changed
                    appender.removeRow(appender.rowCount()-1)
        
        end = appendItem.rowCount()-1
        # loop over all 6 children of scanItem's 'model' 
        for r in range(scanItem.rowCount()):
            sn, sv = scanItem.child(r), scanItem.child(r, 1)
            tn, tv = templateItem.child(r), templateItem.child(r, 1)
            appendChangedParameters(tn, tv, sn, sv, appendItem.child(end))
        
        # move focus to the scan item that has just been inserted
        self.treeView.setCurrentIndex(self.model.index(end, 0, appendItem.index()))
        self.treeView.expandAll()
        
    
    def updateScanParameters(self, item):
        """
        Update the 'scan' models and corresponding xml file when a global 'model' parameter is modified AND
        has NOT been seperately modified before. 
        When a global parameter is changed, the corresponding (already existing) scan child parameters are NOT updated.
        ########
        #REDUNDANT!
        because the template for a scanEditWindow is always taken from the current global 'model' 
        then scan specific items are compared
        
        if self.scansItem.hasChildren() and item.parent().specialType == 0:
            for r in range(self.scansItem.rowCount()):
                scanItem = self.scansItem.child(r)
                if not scanItem.findSpecifiedParameter(Item(isDummy=True), item.name):
                    #
                    # DOES NOT WORK AFTER RELOAD!
                    # 
                    scanModel = scanItem.editModel
                    # search for the parameter in scanModel that has been modified 
                    parameterToChange = scanModel.findItems(item.name, QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
                    # adjust the corresponding valueItem (in the right column!)
                    scanModel.itemFromIndex(scanModel.index(parameterToChange.row(), 1, parameterToChange.parent().index())).value = item.value 
                    #scanModel.writeModelToXML()
        """
    
    
    def newEngine(self):
        """
        Adds an 'engine' child with user-defined parameters.
        Values in the EditWindow() popup are loaded from 'engines_template.xml' and then
        appended to the corresponding 'engine' child.
        """
        model = self.model
        enginesItem = self.enginesItem
        assert enginesItem.name == 'engines'
        
        engineName = "engine_%02i" % (enginesItem.counter+1)
        # select engine name from combobox
        engineType, ok = QtGui.QInputDialog.getItem(self, 'Add engine', 
            'Please select the type of <i>%s</i>:' % engineName, ENGINE_ALIASES, 0, False)
        if ok:
            engineType = unicode(ENGINE_NAMES[ENGINE_ALIASES.index(engineType)])
        else:
            return

        #print engineType
        docID = CSVdoc.name.index(engineType)
        engineItem = Item(name=engineName, docID=docID,specialType=2)
        # Fetch the 'engine' template corresponding to engineType
        commonItem = self.model.findItems(unicode('common'),QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
        templateItem = self.model.findItems(engineType,QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
        #templateItem = self.enginesModel.findItems(engineType, QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
        
        # Manipulate the model of engineWindow. Only one generation of parameters here.
        engineWindow = EditWindow(parent=engineItem)
        
        for daItem in [templateItem,commonItem]:
            for r in range(daItem.rowCount()):
                tn = daItem.child(r).name
                ti =  daItem.child(r).docID 
                nameItem = Item(name=tn,docID=ti)
                valueItem = Item(name=tn,docID=ti, value=daItem.child(r, 1).value)
                engineWindow.model.appendRow([nameItem, valueItem])
        
        if engineWindow.exec_(): # True if OK pressed
            # append 'engine0X' row (== insert at position 'end')
            end = enginesItem.rowCount()
            enginesItem.insertRow(end, [engineItem, Item(name=engineName, isDummy=True)])
            enginesItem.counter += 1
            for row in range(engineItem.editModel.rowCount()):
                nameItem = engineItem.editModel.item(row, 0)
                valueItem = engineItem.editModel.item(row, 1)
                self.appendEngineParameters(nameItem, valueItem, enginesItem.child(end))
            self.contentModified = True
            self.treeView.setCurrentIndex(self.model.index(enginesItem.rowCount()-1, 0, enginesItem.index()))
            self.treeView.expandAll()
        # set current index to something when Cancel pressed
    
    
    def appendEngineParameters(self, nameItem, valueItem, appendItem):
        """
        Appends the parameters from engineWindow to the global 'engines' element in MainWindow.
        """
        if not valueItem.isDummy:
            nn = nameItem.name 
            name = Item(name=nn)
            value = Item(name=nn, value=valueItem.value)
            appendItem.insertRow(appendItem.rowCount(), [name, value])
        else:
            appendItem.insertRow(appendItem.rowCount(), 
                                 [Item(name=nameItem.name), Item(name=nameItem.name, isDummy=True)])
            for n in range(nameItem.rowCount()):
                ni, vi = nameItem.child(n), nameItem.child(n, 1)
                self.appendEngineParameters(ni, vi, appendItem.child(appendItem.rowCount()-1))

            
    def showWarning(self, name, action):
        QtGui.QMessageBox.warning(self, 
            'No \"%s\" element selected!' % name, 
            'Please select a \"%s\" element first. Then click \"%s %s\".' % (name, action, name))
    
    
    def editScan(self):
        """
        Edit the selected 'scan' element, i.e. show the corresponding EditWindow.
        Update entry in MainWindow if any values have been changed.
        """
        model = self.model
        # fetch name of pressed button for distinction
        sender = self.sender()
        buttonName = unicode(sender.text()).split()[1]
        
        if self.treeView.selectionModel().hasSelection():
            index = self.treeView.selectionModel().currentIndex()
            # always edit row corresponding to the item in the left (name) column
            index = model.index(index.row(), 0, index.parent())
            selectedItem = model.itemFromIndex(index)        
            # check if the pressed button agrees with the specialType of the selected item
            if buttonName == 'engine' and selectedItem.specialType == 2:
                #
                # this part is completely REDUNDANT because the engine can be edited in the MainWindow
                # Unless only parameters that differ from defaults are supposed to be shown
                # 
                engineWindow = EditWindow(parent=selectedItem)
                for r in range(selectedItem.rowCount()):
                    tn = selectedItem.child(r).name 
                    nameItem = Item(name=tn)
                    valueItem = Item(name=tn, value=selectedItem.child(r, 1).value)
                    engineWindow.model.appendRow([nameItem, valueItem])
                if engineWindow.exec_():
                    selectedItem.removeRows(0, selectedItem.rowCount())
                    for row in range(selectedItem.editModel.rowCount()):
                        nameItem = selectedItem.editModel.item(row, 0)
                        valueItem = selectedItem.editModel.item(row, 1)
                        self.appendEngineParameters(nameItem, valueItem, self.enginesItem.child(self.enginesItem.rowCount()-1))
                    self.contentModified = True
                    
            elif buttonName == 'scan' and selectedItem.specialType == 1:
                scanWindow = EditWindow(parent=selectedItem) 
                scanTemplatePath = '.temp-scan-template.xml'
                # Load the global 'model' to xml as template for the EditWindow treeView
                # and overwrite the 'value's that are specific for the scan
                # This makes the writing of the scan template to xml obsolete!
                templateItem = self.model.findItems('model', QtCore.Qt.MatchExactly)[0]
                self.writeEditTemplateToXML(scanTemplatePath, templateItem)
                scanWindow.model.setInputPath(scanTemplatePath)
                scanWindow.model.parseXMLtoModel()
                scanWindow.treeView.setLayout()
                if selectedItem.hasChildren():
                    childList = []
                    selectedItem.findYoungestChildren(Item(name=selectedItem.name, isDummy=True), childList)
                    for child in childList:
                        specificItem = scanWindow.model.findItems(child[0], QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
                        specificItem = scanWindow.model.itemFromIndex(scanWindow.model.index(specificItem.row(), 1, specificItem.parent().index()))
                        specificItem.value = child[1]
                        specificItem.setText(child[1])
                if scanWindow.exec_():
                    selectedItem.removeRows(0, selectedItem.rowCount())
                    self.compareScanParameters(model.findItems('model', QtCore.Qt.MatchExactly)[0], 
                        selectedItem.editModel.item(0, 0), self.scansItem)
                    self.contentModified = True
                os.remove(scanTemplatePath)
            else:
                self.showWarning(buttonName, 'Edit')
        else: # no item selected at all
            self.showWarning(buttonName, 'Edit')

            
    def duplicateEngine(self):
        """
        Duplicate the selected 'engine' and append it to the 'engines' children.
        """
        model = self.model
        index = self.treeView.selectionModel().currentIndex()
        # always edit row corresponding to the item in the left (name) column
        index = model.index(index.row(), 0, index.parent())
        selectedItem = model.itemFromIndex(index)
        if selectedItem.specialType == 2:
            enginesItem = self.enginesItem
            engineName = "engine%02i" % (enginesItem.counter+1)
            newEngineItem = Item(name=engineName,docID=enginesItem.docID, specialType=2)
            enginesItem.appendRow([newEngineItem, Item(name=engineName, isDummy=True)])
            for row in range(selectedItem.rowCount()):
                nameItem  = Item(name=selectedItem.child(row).name)
                valueItem = Item(name=selectedItem.child(row, 1).name, value=selectedItem.child(row, 1).value)
                self.appendEngineParameters(nameItem, valueItem, enginesItem.child(enginesItem.rowCount()-1))
            self.contentModified = True
            enginesItem.counter += 1
            self.treeView.setCurrentIndex(model.index(enginesItem.rowCount()-1, 0, enginesItem.index()))
            self.treeView.expandAll()
        else:
            self.showWarning('engine', 'Duplicate')
        
        
    def removeScanOrEngine(self):
        """
        Remove the selected 'scan' or 'engine' element.
        """
        index = self.treeView.selectionModel().currentIndex()
        model = self.model
        # always remove row corresponding to the nameItem
        index = model.index(index.row(), 0, index.parent())
        selectedItem = model.itemFromIndex(index)
        
        # fetch name of pressed button for distinction
        sender = self.sender()
        buttonName = unicode(sender.text()).split()[1]
        
        if selectedItem.specialType > 0:
            # maybe not idiotproof enough, a bit sketchy/clumsy
            if (((buttonName == 'engine') and (selectedItem.specialType == 2)) or 
                ((buttonName == 'scan') and (selectedItem.specialType == 1))):
                selectedItem.parent().removeRow(index.row())
                self.contentModified = True
            else:
                self.showWarning(buttonName, 'Remove')
        else:
            self.showWarning(buttonName, 'Remove')
            
            
    import xml.etree.ElementTree as et
    def convertXMLtoPython(self, inputfile):
        # create the output py file
        outputfile = inputfile.replace('.xml','.py')
        if outputfile==inputfile:
            f = open(PYTHON_OUTPUTPATH, 'w')
        else:
            f = open(outputfile,'w')
        # add the imports
        i = open('imports', 'r')
        iLines = i.readlines()
        i.close()
        for il in iLines:
            f.write(il)
        
        f.write('\n')
        p = 'p'
        f.write('%s = u.Param()\n' % p)
        f.write('\n')
        f.write('### PTYCHO PARAMETERS\n')

        try:
            tree = et.parse(inputfile)
            root = tree.getroot()
        except IOError, err:
            print '%s' % err
            return
        
        # iterate over elements of first generation
        for child in root:
            assert child.tag == 'child'
            self.convertChildren(child, p, f)
            f.write('\n')
        #
        # write some more stuff here, like
        # f.write('P=ptypy.core.Ptycho(p)')
        #
        f.close()

    def convertChildren(self, child, line, f):
        if len(child) == 0:
            name = child.get('name')
            docID = int(child.get('docID'))
            value = str(child.get('value'))
            typ = CSVdoc.type[docID] if docID is not None else None
            quote = '\"' if typ == 'str' and not value=='None' else ''
            # create line of code
            line += '.' + name + ' = ' + quote + value + quote
            # add shortDoc as comment
            line += (50 - len(line))*' ' + '# (%02d) ' % docID + CSVdoc.shortdoc[docID] + '\n'
            f.write(line)
        else:
            line += '.' + child.get('name') 
            f.write(line + ' = u.Param()\n')
            for gchild in child.iterfind('child'):
                self.convertChildren(gchild, line, f)
            f.write('\n')
            
    def loadCSVfile(self):
        appender = self.model
        previous = 3
        for line,name in enumerate(CSVdoc.name):
            #name = CSVdoc.name[line] 
            value = CSVdoc.default[line]
            level = int(CSVdoc.level[line])
            print line,level,name,value
            nameItem = Item(name=name,docID=line)
            # create Items to append
            if len(value) == 0 or name in SPECIAL_ITEMS:
                valueItem = Item(name=name,docID=line, isDummy=True)
                if name in SPECIAL_ITEMS:
                    nameItem.counter = 0
                    #nameItem.counter = int(value) # BE was causing trouble
            else:
                valueItem = Item(name=name,docID=line, value=value)
            # modify the appender if not among siblings
            diff = level - previous
            if diff == -1: # current is child of previous
                appender = appender.child(appender.rowCount()-1)
            elif diff > 0:
                for p in range(diff):
                    appender = appender.parent()
                    if appender == None:
                        appender = self.model
            appender.appendRow([nameItem, valueItem])
            
            """
            if level == 3:
                self.PtypyShortDocs['main'].update({name: row['shortdoc']})
                self.PtypyLongDocs['main'].update({name: row['shortdoc']})
            elif len(value) == 0:
            """
            
            previous = level
    
    def loadCSVfile_legacay(self):
        path = CSV_INPUTPATH
        DELIMITER = '|'
        QUOTECHAR = '"'
        with open(path, 'rb') as csvfile:
            # process with Sniffer first?
            dictreader = csv.DictReader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
            previous = 3
            #parent = 'model'
            appender = self.model
            self.PtypyShortDocs = {'main': {}}
            self.PtypyLongDocs  = {'main': {}}
            for row in dictreader:
                name = row['name']
                value = row['default']
                level = int(row['level'])
                nameItem = Item(name=name)
                # create Items to append
                if len(value) == 0 or name in SPECIAL_ITEMS:
                    valueItem = Item(name=name, isDummy=True)
                    if name in SPECIAL_ITEMS:
                        nameItem.counter = 0
                        #nameItem.counter = int(value) # BE was causing trouble
                else:
                    valueItem = Item(name=name, value=value)
                # modify the appender if not among siblings
                diff = level - previous
                if diff == -1: # current is child of previous
                    appender = appender.child(appender.rowCount()-1)
                elif diff > 0:
                    for p in range(diff):
                        appender = appender.parent()
                        if appender == None:
                            appender = self.model
                appender.appendRow([nameItem, valueItem])
                
                """
                if level == 3:
                    self.PtypyShortDocs['main'].update({name: row['shortdoc']})
                    self.PtypyLongDocs['main'].update({name: row['shortdoc']})
                elif len(value) == 0:
                """
                
                previous = level
                
            # don't display engines' children
            # would be convenient to put them into a model that can be readout later 
            # when a new engine is about to be added
            
            
        

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    window = PtypyMainWindow(app)
    window.show()
    sys.exit(app.exec_())     
