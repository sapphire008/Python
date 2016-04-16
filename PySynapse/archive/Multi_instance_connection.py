"""
//this set it ready to receive a message to a specific chanell, once received, album_slot() will run
   QDBusConnection::sessionBus().connect(QString(),QString(), "open.album", "MY_message", this, SLOT(album_slot(QString)));
//these 3 lines create a signal and send it to DBus
QDBusMessage msg = QDBusMessage::createSignal("/", "open.album", "MY_message");
msg="text"
QDBusConnection::sessionBus().send(msg);
...
...
void album_slot(const QString &text){
if(text=="text") etc
}
"""

#import std
import os, sys

# import stuff for ipc
import getpass, pickle

# Import Qt modules
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import QSharedMemory, QIODevice, SIGNAL
from PyQt4.QtNetwork import QLocalServer, QLocalSocket

class SingletonApp(QApplication):

    timeout = 1000

    def __init__(self, argv, application_id=None):
        QApplication.__init__(self, argv)

        self.socket_filename = unicode(os.path.expanduser("~/.ipc_%s"
                                                          % self.generate_ipc_id()) )
        self.shared_mem = QSharedMemory()
        self.shared_mem.setKey(self.socket_filename)

        if self.shared_mem.attach():
            self.is_running = True
            return

        self.is_running = False
        if not self.shared_mem.create(1):
            print >>sys.stderr, "Unable to create single instance"
            return
        # start local server
        self.server = QLocalServer(self)
        # connect signal for incoming connections
        self.connect(self.server, SIGNAL("newConnection()"), self.receive_message)
        # if socket file exists, delete it
        if os.path.exists(self.socket_filename):
            os.remove(self.socket_filename)
        # listen
        self.server.listen(self.socket_filename)

    def __del__(self):
        self.shared_mem.detach()
        if not self.is_running:
            if os.path.exists(self.socket_filename):
                os.remove(self.socket_filename)


    def generate_ipc_id(self, channel=None):
        if channel is None:
            channel = os.path.basename(sys.argv[0])
        return "%s_%s" % (channel, getpass.getuser())

    def send_message(self, message):
        if not self.is_running:
            raise Exception("Client cannot connect to IPC server. Not running.")
        socket = QLocalSocket(self)
        socket.connectToServer(self.socket_filename, QIODevice.WriteOnly)
        if not socket.waitForConnected(self.timeout):
            raise Exception(str(socket.errorString()))
        socket.write(pickle.dumps(message))
        if not socket.waitForBytesWritten(self.timeout):
            raise Exception(str(socket.errorString()))
        socket.disconnectFromServer()

    def receive_message(self):
        socket = self.server.nextPendingConnection()
        if not socket.waitForReadyRead(self.timeout):
            print >>sys.stderr, socket.errorString()
            return
        byte_array = socket.readAll()
        self.handle_new_message(pickle.loads(str(byte_array)))

    def handle_new_message(self, message):
        print "Received:", message

# Create a class for our main window
class Main(QtGui.QMainWindow):
    def __init__(self):
         QtGui.QMainWindow.__init__(self)
         # This is always the same
         self.ui=Ui_MainWindow()
         self.ui.setupUi(self)

if __name__ == "__main__":
    app = SingletonApp(sys.argv)
    if app.is_running:
        # send arguments to running instance
        app.send_message(sys.argv)
    else:
        MyApp = Main()
        MyApp.show()
        sys.exit(app.exec_())
