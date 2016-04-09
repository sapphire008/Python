Using PyQt 5 with Spyder.
I have a toy project where I want to go the whole scripted/interpreted way with Qt. For that I use this toolchain: Qt 5.1 with QtQuick 2.0, PyQt 5 and Python 3.3. Lately my Python-Editor of choice, Spyder, does support Python 3.3, but it is based on PyQt 4 and a release of Qt 4. That does not fit my bill, as I need some new 5.0 stuff from the QMultimedia module. (If you need just Qt 4 functionality, that toolchain works like a charm and you can stop reading.)

Just downloading a binary release of PyQt 5 and installing it, fails as it detects the PyQt 4, which needs to be uninstalled first, but doing this kills Spyder. Bummer.
Downloading the source and building it (which is actually very straightforward, just follow the instructions in the readme - don't forget to do (n)make install and update the Python module name list (in Spyder)), does the trick. Now both PyQt releases are available.
But importing QtCore from PyQt 5 causes a runtime crash:
RuntimeError: the PyQt4.QtCore and PyQt5.QtCore modules both wrap the QObject class
Obviously PyQt 4 is imported, too. A quick
import sys
sys.modules.keys()​​​
confirms that. The reason for that is that Spyder installs a special input hook, replacing the input hook of PyQt, which is said to cause problems with Spyder. Fortunately you can disable this hook under Tools > Preferences > Console > External modules. After a restart of the current console, you can now import QtCore and you can start developing your stuff.
There is a slight annoying thing regarding the hook though: you now can not interactively work with your application in the console. The solution for that comes in the next post.
