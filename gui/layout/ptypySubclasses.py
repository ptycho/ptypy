#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt4 import QtXml, QtGui, QtCore


ENGINE_ALIASES = ['Difference Map', 'ePIE', 'Maximum Likelihood']
ENGINE_NAMES = ['DM', 'EPIE','ML']
SPECIAL_ITEMS = ['scans', 'engines']
CSV_INPUTPATH = 'NEW-ptypy-template.csv'


class Item(QtGui.QStandardItem):
    """
    Custom item class. When initialising, at least a name has to be given.
    You can create either
    - a nameItem  (attributes: name) or
    - a valueItem (attributes: name, value) or
    - a dummyItem (attributes: name, isDummy=True) for parent items in value column or
    - a scanItem  (attributes: name, specialType=1)
    - an engineItem (attributes: name, specialType=2)
    - a scans/enginesItem (attributes: name, counter)
    """
    def __init__(self, name='', docID=None, value=None, isDummy=False, specialType=0):
        super(Item, self).__init__()
        self.name = name 
        self.value = value
        try:
            self.docID = docID if docID is not None else CSVdoc.name.index(name)
        except ValueError:
            self.docID = None
            
        self.isDummy = isDummy
        self.specialType = specialType
        self.editModel = None   # stores model of EditWindow()
        self.counter = None # counts the nr of scans and engines, resp.
        self.setEditable(False)
        
        if not self.isDummy:
            if value is None: # any Item except valueItem
                self.setText(name)
            else: 
                self.setText(value)
                if self.name != 'name': # Engine type name is not supposed to be changed
                    self.setEditable(True)
        if self.specialType > 0:
            # enable renaming of scanItem and engineItem
            self.setEditable(True)

    
    def writeChildrenToXML(self, valueItem, xmlWriter, isScanTemplate=False):
        """
        Children of the nameItem 'self' are written to xml file using xmlWriter.
        The corresponding valueItem needs to be passed as well.
        """
        xmlWriter.writeStartElement('child')
        xmlWriter.writeAttribute('name', self.name)
        xmlWriter.writeAttribute('docID',str(self.docID))
        
        if not valueItem.isDummy:    
            xmlWriter.writeAttribute('value', valueItem.value)
        else: # element is a parent
            if self.specialType > 0:
                xmlWriter.writeAttribute('specialType', str(self.specialType))
            if self.counter is not None:      # scansItem or enginesItem
                xmlWriter.writeAttribute('counter', str(self.counter))
            for r in range(self.rowCount()):
                name, value  = self.child(r, 0), self.child(r, 1)
                name.writeChildrenToXML(value, xmlWriter, isScanTemplate=isScanTemplate)
                
        xmlWriter.writeEndElement()
            
            
    def findSpecifiedParameter(self, valueItem, nameToFind):
        """
        Helper function that is evaluated when global 'model' parameters of MainWindow are modified.
        """
        if not valueItem.isDummy:
            return self.name == nameToFind
        else:
            for r in range(self.rowCount()):
                name, value  = self.child(r, 0), self.child(r, 1)
                if name.findSpecifiedParameter(value, nameToFind):
                    return True
                    
    def findYoungestChildren(self, valueItem, childList):
        """
        Helper function that returns a list with the child items of a scanItem.
        Used to compare with the global 'model' when an EditWindow is about to be opened.
        """
        if not valueItem.isDummy:
            childList.append((valueItem.name, valueItem.value))
        else:
            for r in range(self.rowCount()):
                name, value  = self.child(r, 0), self.child(r, 1)
                name.findYoungestChildren(value, childList)



from Ui_EditWindow import Ui_Dialog
class EditWindow(QtGui.QDialog, Ui_Dialog):
    """
    Window that pops up when a scan or engine is added or edited.
    Parent is the corresponding scan/engineItem.
    The corresponding model is never supposed to be set up at initialising but is separately 
    read in from the global 'model' (in case of 'scan') or the already existing engineItem.
    """
    def __init__(self, parent=None):
        super(EditWindow, self).__init__()
        self.parent = parent 
        self.setupUi(self)
        self.setWindowTitle('Edit %s' % parent.name)

        self.model = PtypyModel(setupAtInit=False)
        self.treeView.setModel(self.model)
        self.treeView.setLayout()

        # connections
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.model.itemChanged.connect(self.treeView.modifyValue)

    
    def accept(self):
        """
        Reimplementation, writes model of current 'scan'/'engine' to parent.model
        Necessary for comparison with global 'model' parameters.
        """
        self.parent.editModel = self.model
        QtGui.QDialog.accept(self)
        

        
import xml.etree.ElementTree as et
class PtypyModel(QtGui.QStandardItemModel):
    """
    Custom Model class.
    Capable of parsing xml file to model and write model to xml file.
    """
    def __init__(self, inputPath="", outputPath="", parent=None, setupAtInit=True):
        super(PtypyModel, self).__init__()
        self.parent = parent # redundant
        self.__inputPath = inputPath
        self.__outputPath = outputPath
        self.setHorizontalHeaderLabels(['Name', 'Value'])
        if setupAtInit:
            self.parseXMLtoModel()
        
    def setInputPath(self, path):
        """
        Called when file is loaded by user. 
        """
        self.__inputPath = path 
        
    
    def setOutputPath(self, path):
        """
        Called before model is written to xml file.
        """
        self.__outputPath = path 
        
        
    def getOutputPath(self):
        return self.__outputPath
        
    
    def child(self, row):
        return self.item(row)
    
        
    def parseXMLtoModel(self):
        """
        Reads in xml file given via inputPath and parses it recursively to model.
        Uses xml.etree.ElementTree to fetch the xml elements.
        """
        def setChildren(child, appender):
            # deepest possible branch reached, element has no children
            childName = child.get('name')
            docID = int(child.get('docID'))#,CSVdoc.name.index(childName)))
            if len(child) == 0:
                nameItem = Item(name=childName,docID=docID)
                # this is only true when a template model with childless 'scans' or 'engines' item is loaded
                if childName in SPECIAL_ITEMS:
                    nameItem.counter = int(child.get('counter'))
                    valueItem = Item(name=childName,docID=docID, isDummy=True)
                else:
                    valueItem = Item(name=childName,docID=docID, value=child.get('value'))
                appender.appendRow([nameItem, valueItem])
            else:
                if 'specialType' in child.attrib.keys():
                    nameItem = Item(name=childName,docID=docID, specialType=int(child.get('specialType')))
                else:
                    nameItem = Item(name=childName,docID=docID)
                if childName in SPECIAL_ITEMS:
                    nameItem.counter = int(child.get('counter'))
                # place a uneditable dummy in second column if element has children
                appender.appendRow([nameItem, Item(name=childName,docID=docID, isDummy=True)])
                
                for gchild in child.iterfind('child'):
                    setChildren(gchild, nameItem)

        try:
            tree = et.parse(self.__inputPath)
            root = tree.getroot()
        except IOError, err:
            print "Oopsie, something went wrong when parsing from xml! %s" % err
            return
        
        # iterate over children of 1st generation
        for child in root:
            assert child.tag == 'child'
            setChildren(child, self)
            
    
    def writeModelToXML(self):
        """
        Writes the current model to xml file 'outputPath' ending using recursion.
        """
        xmlWriter = QtCore.QXmlStreamWriter()
        # do indents and paragraphs in xml output
        xmlWriter.setAutoFormatting(True) 
        xmlFile = QtCore.QFile(self.__outputPath)
        print "written to " + self.__outputPath
        
        if (xmlFile.open(QtCore.QIODevice.WriteOnly) == False):
            QtGui.QMessageBox.warning(self, "Error!", "Error opening file")
        else:
            xmlWriter.setDevice(xmlFile)
            xmlWriter.writeStartDocument()
            xmlWriter.writeStartElement('root')

            for row in range(self.rowCount()):
                nameItem, valueItem = self.item(row), self.item(row, 1)
                nameItem.writeChildrenToXML(valueItem, xmlWriter)
            xmlWriter.writeEndElement()
            xmlWriter.writeEndDocument()



class PtypyTreeView(QtGui.QTreeView):
    """
    Custom TreeView class.
    Includes a type check method when entering new values and
    a setLayout() method for convenience.
    """
    def __init__(self, parent=None):
        super(PtypyTreeView, self).__init__()
        self.parent = parent #redundant
        self.contentModified = False
        self.incorrectValueEntered = False
        self.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked|
                             QtGui.QAbstractItemView.SelectedClicked)
                             #|QtGui.QAbstractItemView.AnyKeyPressed)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.searchScanContextMenu)


    def searchScanContextMenu(self, point):
        """
        Opens a context menu when the user right-clicks on the value (== right) column 
        of the 'file' parameter.
        """
        self.contextMenu = QtGui.QMenu(self)
        self.addScanFileAction = QtGui.QAction('Search for scan file...', self.contextMenu)
        self.contextMenu.addAction(self.addScanFileAction)
        self.addScanFileAction.triggered.connect(self.showLoadScanDialog)
        
        self.fileItem = self.model().itemFromIndex(self.indexAt(point))
        if self.fileItem.name == 'source' and self.fileItem.column() == 1:
            self.contextMenu.exec_(self.mapToGlobal(point))


    def showLoadScanDialog(self):
        """
        Opens a file dialog so that the user can look for the h5 scan path.
        """
        loadFileName = QtGui.QFileDialog.getOpenFileName(self, 'Load h5 scan file',
            QtCore.QDir.currentPath(), 'h5 file (*.h5)')
        if not loadFileName.isNull():
            loadFileName = unicode(loadFileName)
            self.fileItem.value = loadFileName
            self.fileItem.setText(loadFileName)
            
        
    def setLayout(self):
        """
        Wraps up some minor layout issues.
        """
        #self.expandAll()
        self.setAnimated(True)
        # automatically resize column width when nodes are expanded or closed
        self.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(True)
            
    
    def hideFileAndSourcetypeRows(self):
        """
        The parameters 'file' and 'sourcetype' are included in 'ptypy_template.xml'
        but not supposed to be displayed as children of 'model' in the MainWindow.
        They can be edited in the EditWindow though.
        """
        model = self.model()
        modelItem = model.findItems('model', QtCore.Qt.MatchExactly)[0]
        fileItem = model.findItems('file', QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
        sourcetypeItem = model.findItems('sourcetype', QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)[0]
        self.setRowHidden(fileItem.row(), modelItem.index(), True)
        self.setRowHidden(sourcetypeItem.row(), modelItem.index(), True)
        
        
    def modifyValue(self, item):
        """
        Checks if the newly entered value of a parameter is of appropriate data type
        """
        currentIndex = self.model().indexFromItem(item)
        value = str(item.text())
        if len(value) == 0:
            QtGui.QMessageBox.warning(self, 
            'No value given',
            'The value field of the parameter <i>%s</i> is empty. Please insert a valid value.' % item.name)
            item.setText(item.value)
            self.setCurrentIndex(currentIndex)
            return 
        
        # handle the renaming of scanItems and engineItems
        #
        # maybe check that first character is not a digit
        #
        if item.specialType > 0:
            item.name = value
            self.contentModified = True
            return
        
        if value.upper() == 'NONE':
            item.value = value
            self.contentModified = True
            return
        
        typ = None
        
        if value.isdigit():
            typ = int
        elif value[0].isdigit():
            typ = float
        elif value[0] == '(':
            typ = tuple
        elif value[0] == '[':
            typ = list
        elif value in ('True', 'False'):
            typ = bool
        else:
            typ = str
        
        # throw annotation if type does not match
        typ = str(typ)[7:str(typ).find("\'", 7)] # fetches actual type from "<type 'int'>"
        supposedTyp = CSVdoc.type[item.docID]
        if typ != supposedTyp:
            QtGui.QMessageBox.warning(self, 
            'Type mismatch', 
            u'The value of the parameter <b>%s</b> is supposed to be of type <i>%s</i>, not <i>%s</i>.\
            Please enter a correct value.' % (item.name, supposedTyp, typ))
            # flag because modifyValue() is called again albeit value is reset and no changes made
            self.incorrectValueEntered = True
            item.setText(item.value)
            self.setCurrentIndex(currentIndex)
        else:
            #
            # add functionality to get the limits
            #ll = PtypyDict.lowerLimits.get(item.name)
            #
            item.value = value 
            if not self.incorrectValueEntered:
                # this flag will be evaluated when saveFile() in MainWindow is executed
                self.contentModified = True
            
            
class PtypyDictionaries():
    """
    A container for lower/upperlimits, short/longDocs and type values that 
    have to be globally accessible.
    Read in from csv file.
    """
    def __init__(self, csv_input_path= 'ptypy_template.csv'):
        self.shortDocs = {}
        self.longDocs = {}
        self.types = {}
        self.lowerLimits = {}
        self.upperLimits = {}
        self.csvInputPath = csv_input_path
        self.loadCSVtoDocs(self.csvInputPath)
        
    def loadCSVtoDocs(self, path):
        import csv
        DELIMITER = '|'
        QUOTECHAR = '"'
        with open(path, 'rb') as csvfile:
            # process with Sniffer first?
            dictreader = csv.DictReader(csvfile, delimiter=DELIMITER, quotechar=QUOTECHAR)
            # convert to dictionaries
            for row in dictreader:
                self.shortDocs.update({row['name']: row['shortdoc']})
                self.longDocs.update({row['name']: row['longdoc']})
                self.types.update({row['name']: row['type']})
                self.lowerLimits.update({row['name']: row['lowerlimit']})
                self.upperLimits.update({row['name']: row['upperlimit']})


class CSVcontent():
    """
    we will be using lists instead of dictionaries to avoid ambiguity 
    in parameter names (they  may appear more than once)
    """
    def __init__(self, csv_input_path= 'ptypy_template.csv',delimiter = '|',quotechar = '"',**kwargs):
        #self.delimiter = delimiter
        #self.quotechar = quotechar
        #self.path = csv_input_path
        self.csvdict = self._loadcsv(csv_input_path,delimiter=delimiter,quotechar=quotechar,**kwargs)
    
    def _loadcsv(self,path,**kwargs):
        import csv
        print "loading default from " + path
        with open(path, 'rb') as csvfile:
            dictreader = csv.DictReader(csvfile, **kwargs)
            # convert to dictionaries
            fields = dictreader.fieldnames
            dct = dict([(key,[]) for key in fields])
            for row in dictreader:
                for field in fields:
                    dct[field].append(row[field])
                    
        for field in fields:
            self.__dict__[field]=dct[field]
        
        return dct
# needs to be globally accessible
CSVdoc = CSVcontent(CSV_INPUTPATH)
