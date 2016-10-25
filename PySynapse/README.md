# README for new Synapse written in Python 3 ##
Script structure

* `SynapseQt.py`: main window
* `app`: other apps / windows
  - `app/Scope.py`: window for trace display
  - `app/AccordionWidget.py`: a class for designing side dock panel toolbox
  - `app/Mirage.py`: window for image display

* `util`: utility functions
  - `util/ImportData`: data reading utilities
  - `util/Database`: database function (To be implemented)

* `resources`: assets, icons, fonts, etc

**Planned features of Scope window**
* ~~Curve fitting~~
* Extracellular spike detection
* ~~Trace subtraction, average, ... arithmetic manipulation~~

**Planned features of Mirage window**
* Display a stack as movie
* Display the Maximum Pixel Intensity image
* dF/F trace

#########################################################################
1. Dependencies:
    - numpy
    - pandas
    - PyQt4
    - pyqtgraph: for data display (trace and image)
    - matplotlib: for exporting figures

2. To-dos:
    - ~~Icons in file browser~~
    - ~~Table view model for sequence listing~~
      - ~~allow update of table columns via menu bar Episodes --> columns --> checklist --> update button~~
      - ~~columns are header information of the data~~
    - ~~pyqtgraph for traces~~
    - ~~View layout configuration: channels x streams of data to display~~
    - ~~Range selection / cursor tool~~: for event detection and data analyses
      * ~~Initial and end values, end - initial difference~~
      * max, min, average, median, std
      * ~~EPSP, IPSP, EPSC, IPSC, action potentials~~, extracellular spikes
      * ~~exponential / double exponential curve fitting~~
    - Indexing / annotation system. Load all the meta info of the data files into a database. Allow the user to search for keywords or key values.
    - ~~Average trace display~~

## Update Sep 29, 2016
* Added functionality to export traces arranged horizontally; good for experiments acquired over several episodes and to be viewed as a whole --> will also add horizontal arrangement for pyqtgraph Scope window as well in the future.
* Added setting options specific to horizontal plot exports --> but need to group them better in the future.

## Update Sep 10, 2016
* Added settings window. Default settings are saved under ./resouces/config.ini and interfaced by GUI under the main synapse window: File/Settings.
  - For now, implemented settings for exporting traces. Planned to extend the new settings function to other aspects of the program
  - Extended some functions in trace export
* Set icons for each app window
* Corrected some typos

## Update Aug 20, 2016
* Added arithmetic tool to calculate traces (averages, subtractions)
* Added curve fitting tools to fit polynomials, exponentials (3 equations), power law (2 equations)
* Fixed some bugs on exporting figures
* Changed to PySynapse ver 0.3

## Update Jun 11, 2016
* Added "Arithmetic", "Layout", and "Event" tools in the toolbox
  - "Arithmetic": remove baseline ("null" checkbox) and trace averaging / manipulation (to be implemented)
  - "Layout": add and remove data streams
  - "Event": event detector, including APs, PSPs. Extracellular spike detection is yet to be implemented
* Fixed various bugs and fine tuned some behaviors.

## Update Apr 24, 2016
* Side dock panel toolbox
  - Added "Channel" toolbox. Now can add and remove data channel at ease.
  - Started "Analysis" toolbox. Need to write the corresponding analysis functions first.

## Update Apr 9, 2016
* Export multiple traces using matplotlib; only 'overlap' configuration is fully working
* Exported traces can be color coded, if the user turn on "colorfy" option in the Scope window; Colors cycles use tableau10
* Files are exported as .eps, and font set to Helvetica, using a .ttf file under `./resources`; fontsize=12.0. This should be cross-platform, as it does not depend on the system's font repository. It should also be editable in vector graphics editor. From experience, if using 'Arial' font in Windows, the font header in the .eps file cannot be recognized by editors, and the entire graphics cannot be imported successfully (true for InkScape and CorelDraw)

## Update Mar 20, 2016
* Improved file system interface
  - Folders and files with numbers now sort intuitively to human reading
  - Fixed network drive volume info query
  - Fixed file system horizontal scroll
* Scope window:
  - Allow setting view range of traces
* Exporting:
  - Allow exporting the traces to a .eps file. For now, can only export from the same cell. A restructure would be needed for other cells.
  - Yet to implement the actual plotting and exporting part. Has set up the hooks.

## Update Mar 13, 2016
* Scope window is fully functional now.
  - Plot traces with or without color. Colors usage are tracked correctly. Colors are drawn from tableau10.
  - Plot traces of multiple channels of data, with time domain linked
  - Plot traces of multiple episodes of data, correctly distribute them across channels of data
* Added toggle functionality in both main and scope windows
  - Allow toggle of additional columns of episode list tableview
  - Allow toggle of side panel of scope window
  - Allow toggle between colored traces and black traces

## Update Mar 6, 2016
* The table view is fully functional. Clicking on the table selects a row. Multiple rows can be selected by dragging along the rows, by clicking while holding Ctrl, or by holding SHIFT.
* Each selection will highlight with a preferred, custom, blue color.
* Clicking the episode will spin up the Scope window. By tracking the history of clicking (from the previous state), it is possible to load up the traces faster.

## Update Feb 20, 2016
* Now in Windows system, at startup, the program will list all the drives more efficiently via wmic
  * In this implementation, I addressed potentially problematic X:\ drives. When X:\ was mounted but disconnected due to network problem, the Windows system still register it as an active drive, but unable to get volume information. It will takes 10s of seconds before it return an error. With this implementation, I set out a timeout of 2 seconds from subprocess calling to inquire volume name information. Upon encountering disconnected X:\ drive, wmic volume will return as not available very quickly. To further safeguard and reduce startup time, if wmic call takes more than 2 seconds to inquire volume name, it will give up the inquiry.

## Update Feb 13, 2016
* Changed to Version PySynapse 0.2
* Reimplemented a custom FileSystemModel to allow insertion of custom rows, using QAbstractitemModel. Original attempt using QFileSystemModel was unsuccessful and complicated.
* Default startup directory depends on operating system.
  * Windows: list all the drives, like default QFileSystemModel
  * Mac: /Volumes
  * Linux: /
