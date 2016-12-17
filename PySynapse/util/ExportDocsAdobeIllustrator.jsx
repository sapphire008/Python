/**********************************************************

ADOBE SYSTEMS INCORPORATED
Copyright 2005 Adobe Systems Incorporated
All Rights Reserved

NOTICE:  Adobe permits you to use, modify, and
distribute this file in accordance with the terms
of the Adobe license agreement accompanying it.
If you have received this file from a source
other than Adobe, then your use, modification,
or distribution of it requires the prior
written permission of Adobe.

*********************************************************/

/**********************************************************

ExportDocsAsPNG24.js

DESCRIPTION

This sample gets files specified by the user from the
selected folder and batch processes them and saves them
as PDFs in the user desired destination with the same
file name.

**********************************************************/

// Main Code [Execution of script begins here]

// uncomment to suppress Illustrator warning dialogs
// app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;


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
// exportFigures_AI_CS6(sourceFile="C:/Users/Edward/Desktop/Figure 1-6 Percent Persistence 2.svg", targetFile="C:/Users/Edward/Desktop/Figure 1-6 Percent Persistence 2.eps", exportType='eps', ExportOpts=null)
exportFigures_AI_CS6(sourceFile=arguments[0], targetFile=arguments[1], exportType=arguments[2])
// print(arguments[0])
// print(arguments[1])
// print(arguments[2])
