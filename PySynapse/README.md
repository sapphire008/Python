## README for new Synapse written in Python ##
Script structure

* SynapseQt.py: main window
* app: other apps / windows
  - app/Scope.py: window for trace display
  - app/Imreg.py: window for image display

* util: utility functions
  - util/ImportData: data reading utilities
  - util/Database: database function (To be implemented)
  - util/Analyzer: simple data analyzers (To be implemented)


#########################################################################
1. Dependencies:
    - numpy
    - pandas
    - pyqtgraph: for data display (trace and image)
    - matplotlib: for exporting figures

2. To-dos:
    - Icons in file browswer
    - Table view model for sequence listing
      - allow update of table columns via menu bar Episodes --> columns --> checklist --> update button
      - columns are header information of the data
    - pyqtgraph for traces
    - Indexing system. Load all the meta info of the data files into a database. Allow the user to search for keywords or key values.


## Update Feb 13, 2016
* Changed to Version PySynapse 0.2
* Reimplemented a custom FileSystemModel to allow insertion of custom rows, using QAbstractitemModel. Original attempt using QFileSystemModel was unsuccessful and complicated.
* Default startup directory depends on operating system.
  * Windows: list all the drives, like default QFileSystemModel
  * Mac: /Volumes
  * Linux: /

## Update Feb 20, 2016
* Now in Windows system, at startup, the program will list all the drives more efficiently via wmic
  * In this implementation, I addressed potentially problematic X:\ drives. When X:\ was mounted but disconnected due to network problem, the Windows system still register it as an active drive, but unable to get volume information. It will takes 10s of seconds before it return an error. With this implementation, I set out a timeout of 2 seconds from subprocess calling to inquire volume name information. Upon encountering disconnected X:\ drive, wmic volume will return as not available very quickly. To further safeguard and reduce startup time, if wmic call takes more than 2 seconds to inquire volume name, it will give up the inquiry.

## Update Mar 6, 2016
* The table view is fully functional. Clicking on the table selects a row. Multiple rows can be selected by dragging along the rows, by clicking while holding Ctrl, or by holding SHIFT.
* Each selection will highlight with a preferred, custom, blue color.
* Clicking the episode will spin up the Scope window. By tracking the history of clicking (from the previous state), it is possible to load up the traces faster.
