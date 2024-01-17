# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:44:36 2016

Call Adobe Illustrator to convert .svg to .eps

@author: Edward
"""
import os
import signal
import subprocess
import time
from pdb import set_trace

jsx_file_str_AI_CS6 = """
function exportFigures_AI_CS6(sourceFile, targetFile, exportType, ExportOpts) {
  if (sourceFile){ // if not an empty string
    var fileRef = new File(sourceFile)
    var sourceDoc = app.open(fileRef); // returns the document object
  } else { // for empty string, use current active document
      sourceDoc = app.activeDocument();
  }
  var newFile = new File(targetFile) // newly saved file

  switch(exportType){
     case 'png':
       if (ExportOpts == null) {
          var ExportOpts = new ExportOptionsPNG24()
          ExportOpts.antiAliasing = true;
          ExportOpts.transparency = true;
          ExportOpts.saveAsHTML = true;
        }
       // Export as PNG
       sourceDoc.exportFile(newFile, ExportType.PNG24, ExportOpts);
     case 'tiff':
       if (ExportOpts == null) {
          var ExportOpts = new ExportOptionsTIFF();
          ExportOpts.resolution = 600;
          ExportOpts.byteOrder = TIFFByteOrder.IBMPC;
          ExportOpts.IZWCompression = false;
          ExportOpts.antiAliasing = true
        }
       sourceDoc.exportFile(newFile, ExportType.TIFF, ExportOpts);
     case 'svg':
       if (ExportOpts == null) {
          var ExportOpts = new ExportOptionsSVG();
          ExportOpts.embedRasterImages = true;
          ExportOpts.embedAllFonts = true;
          ExportOpts.fontSubsetting = SVGFontSubsetting.GLYPHSUSED;
        }
       // Export as SVG
       sourceDoc.exportFile(newFile, ExportType.SVG, ExportOpts);
     case 'eps':
       if (ExportOpts == null) {
          var ExportOpts =  new EPSSaveOptions();
          ExportOpts.cmykPostScript = true;
          ExportOpts.embedAllFonts = true;
        }
       // Export as EPS
       sourceDoc.saveAs(newFile, ExportOpts);
  }
  // Close the file after saving. Simply save another copy, do not overwrite
  sourceDoc.close(SaveOptions.DONOTSAVECHANGES);
}

// Use the function to convert the files
exportFigures_AI_CS6(sourceFile="{format_source_file}", targetFile="{format_target_file}", exportType="eps", ExportOpts=null)
// exportFigures_AI_CS6(sourceFile=arguments[0], targetFile=arguments[1], exportType=arguments[2])
"""


def svg2eps_ai(source_file, target_file, \
               illustrator_path="D:/Edward/Software/AdobeIllustratorCS6(64bit)Portable/Support Files/Contents/Windows/Illustrator.exe",\
               jsx_file_str = jsx_file_str_AI_CS6, DEBUG=False):
    """Use Adobe Illustrator to convert svg to eps"""
    # Change the strings
    jsx_file_str = jsx_file_str.replace('{format_source_file}', source_file)
    jsx_file_str = jsx_file_str.replace('{format_target_file}', target_file).replace('\\','/')
    tmp_f = os.path.abspath(os.path.join(os.path.dirname(target_file), "tmp.jsx"))
    # Write to a temporary location
    f = open(tmp_f, 'w')
    f.write(jsx_file_str)
    f.close()

    # Remove previous target file if already existed
    if os.path.isfile(target_file):
        os.remove(target_file)

    if sys.platform == "win32":
        # subprocess.check_call([illustrator_path, '-run', tmp_f])
        cmd = " ".join(['"'+illustrator_path+'"', '-run', '"'+tmp_f+'"'])
        # Run
        pro = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        
    elif sys.platform == "darwin": # use osasscript
        cmd = f"""osascript -e 'tell application "Adobe Illustrator" to do javascript (file "{tmp_f}")'"""
        pro = subprocess.call(cmd, shell=True)
    
    print(pro.stdout)
    # continuously check if new files are updated
    time.sleep(5.0)
    sleep_iter = 5.0
    max_sleep_iter = 40
    while not os.path.isfile(target_file):
        time.sleep(1.0)
        sleep_iter = sleep_iter + 1.0
        if sleep_iter > max_sleep_iter:
            break

    # pro.terminate()
    #os.kill(os.getpid(), signal.SIGTERM)  # Send the signal to all the process groups
    pro.kill()
    os.remove(tmp_f)

def svg2eps_inkscape(source_file, target_file, \
                     inkscape_path='"D:\\Edward\\Software\\inkscape-0.91-1-win64\\inkscape.exe"'):
    """Use inkscape to convert svg to eps"""
    # cmd = "inkscape in.svg -E out.eps --export-ignore-filters --export-ps-level=3"
    cmd = inkscape_path+" "+source_file+" --export-eps="+target_file +" --export-ignore-filters --export-ps-level=3"
    print(cmd) # Problem: text was not kept as text, but converted into paths
    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    #subprocess.check_call([inkscape_path, source_file, '-E', target_file])
    print(pro.stdout)
    
#def svg2eps_cloudconvert(source_file, target_file):
#    import cloudconvert
#    api = cloudconvert.Api('5PGyLT7eAn0yLbnBU3G-7j1JLFWTfcnFUk6x7k_lhuwzioGwqO7bVQ-lJNunsDkrr9fL1JDdjdVog6iDZ31yIw')
#    process = api.convert({"input": "upload",
#                           "file": open('R:/temp.svg', 'rb'),
#                           "inputformat": "svg",
#                           "outputformat": "eps",
#                           })
#    process.wait()
#    process.download()




if __name__ == '__main__':
    source_file = '"R:\\temp.svg"'
    target_file = '"R:\\temp.eps"'
    illustrator_path="D:/Edward/Software/Adobe Illustrator CS6/Support Files/Contents/Windows/Illustrator.exe"
    javascript_path="D:\\Edward\\Documents\\Assignments\\Scripts\\Python\\PySynapse\\util\\ExportDocsAdobeIllustrator.jsx"
    # svg2eps_ai(source_file, target_file)
    svg2eps_inkscape(source_file, target_file)
