# Initialize the layout widget
       widgetFrame = QtGui.QFrame(self)
       widgetFrame.setLayout(QtGui.QGridLayout())
       widgetFrame.layout().setSpacing(10)
       widgetFrame.setObjectName(_fromUtf8("LayoutWidgetFrame"))
       all_streams = sorted(set([l[0] for l in all_layouts]))
       all_streams = [s for s in ['Voltage', 'Current','Stimulus'] if s in all_streams]
       all_channels = sorted(set([l[1] for l in all_layouts]))
       # Layout setting table
       self.setLayoutTable(all_streams, all_channels)
       # Buttons for adding and removing channels and streams
       addButton = QtGui.QPushButton("Add") # Add a channel
       addButton.clicked.connect(lambda: self.addLayoutRow(all_streams=all_streams, all_channels=all_channels))
       removeButton = QtGui.QPushButton("Remove") # Remove a channel
       removeButton.clicked.connect(self.removeLayoutRow)
       # Add the exisiting channels and streams to the table
       widgetFrame.layout().addWidget(addButton, 1, 0)
       widgetFrame.layout().addWidget(removeButton, 1, 1)
       widgetFrame.layout().addWidget(self.layout_table, 2, 0, self.layout_table.rowCount(), 2)
       return widgetFrame
