#!/usr/bin/env python

import sys
from PyQt5.QtWidgets import QMessageBox, QApplication

class Example(QMessageBox):
    
    def __init__(self):
        super(Example, self).__init__()
        
        self.initUI()
        
    def initUI(self):      

        self.setIcon(self.Information)
        self.setText("Excenter finished!")
        self.setInformativeText("You may now save the determined position and define acquisition.")
        self.setWindowTitle("Excenter finished!")
        
        #self.btn = QtGui.QPushButton('OK', self)
        #self.btn.move(50, 50)
        #self.btn.clicked.connect(self.exit)
        
        #self.label = QtGui.QLabel(self)
        #self.label.setText('Excenter finished!')
        #self.label.move(22, 22)
        
        #self.setGeometry(100, 100, 150, 150)
        #self.setWindowTitle('Excenter finished!')
        self.show()
                
def main():
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
