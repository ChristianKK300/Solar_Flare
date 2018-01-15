import numpy as np
from matplotlib import pyplot as plt
from lu_ham import LuHamilton
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

automate = LuHamilton((20, 20))

class MyView(pg.GraphicsWindow):
    def __init__(self):
        super(MyView, self).__init__()
        l = pg.GraphicsLayout(border=(100,100,100))
        self.setCentralWidget(l)
        self.show()
        self.setWindowTitle('pyqtgraph example: GraphicsLayout')
        self.resize(800,600)
        vb = l.addViewBox(lockAspect=True)
        self.img = pg.ImageItem()
        vb.addItem(self.img)
        l.nextRow()
        text = 'state of the network'
        self.label = l.addLabel(text, colspan=2)

        # self.sp_state = np.zeros(poppy.sp.getColumnDimensions(), dtype="uint32")
        # self.prev_sp_state = np.zeros(poppy.brain.L23_shape, dtype="uint32")
        # self.average_difference = 0.4
        # self.data = np.zeros(100)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)
        self.counter = 0

    def update(self):
        self.counter += 1
        automate.evolve()
        self.img.setImage(automate.cells/np.max(automate.cells) * 255)

        output_text = 'Iteration: ' + str(self.counter)
        self.label.setText(output_text)




app = QtGui.QApplication([])
view = MyView()

app.exec_()