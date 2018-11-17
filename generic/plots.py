# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 02:21:18 2016

General utilities for plotting

@author: Edward
"""
import sys
import os
import numpy as np


import os
import signal
import subprocess
import time
from pdb import set_trace
from scipy import stats

import matplotlib as mpl
# mpl.use('PS')
#mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
#mpl.rcParams['ps.useafm'] = True
#mpl.rcParams['pdf.use14corefonts'] = True
#mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# some predetermined parameters
fontsize = {'title':10, 'xlab':8, 'ylab':8, 'xtick':5,'ytick':5,'texts':5,
            'legend': 5, 'legendtitle':6, 'xminortick':5, 'yminortick':5, "colorbartick": 5} # font size

# unit in points. This corresponds to 0.25mm (1pt  = 1/72 inch)
bar_line_property = {'border': 0.70866144, 'h_err_bar': 0.70866144, 'v_err_bar': 0.70866144, 
                     'xaxis_tick': 0.70866144, 'yaxis_tick': 0.70866144, 'xaxis_spine': 0.70866144,
                     'yaxis_spine': 0.70866144} # in mm.


def SetFont(ax, fig, fontsize=12,fontname='Arial',items=None):
        """Change font properties of all axes
        ax: which axis or axes to change the font. Default all axis in current
            instance. To skip axis, input as [].
        fig: figure handle to change the font (text in figure, not in axis).
        Default is any text items in current instance. To skip, input as [].
        fontsize: size of the font, specified in the global variable
        fontname: fullpath of the font, specified in the global variable
        items: select a list of items to change font. ['title', 'xlab','ylab',
               'xtick','ytick', 'texts','legend','legendtitle']
       
        """
        
        def get_ax_items(ax):
            """Parse axis items"""
            itemDict={'title':[ax.title], 'xlab':[ax.xaxis.label],
                    'ylab':[ax.yaxis.label], 'xtick':ax.get_xticklabels(),
                    'ytick':ax.get_yticklabels(),
                    'xminortick': ax.get_xminorticklabels(),
                    'yminortick': ax.get_yminorticklabels(),
                    'texts':ax.texts if isinstance(ax.texts,(np.ndarray,list))
                                         else [ax.texts],
                    'legend': [] if ax.legend_ is None
                                        else ax.legend_.get_texts(),
                    'legendtitle':[] if ax.legend_ is None
                                        else [ax.legend_.get_title()]
                                        }
            itemList, keyList = [], []
            if items is None: # get all items
                for k, v in iter(itemDict.items()):
                    itemList += v
                    keyList += [k]*len(v)
            else: # get only specified item
                for k in items:
                    itemList += itemDict[k] # add only specified in items
                    keyList += [k]*len(itemDict[k])
            
            return(itemList, keyList)
            
        def get_fig_items(fig):
            """Parse figure text items"""
            itemList = fig.texts if isinstance(fig.texts,(np.ndarray,list)) \
                                    else [fig.texts]
            keyList = ['texts'] * len(itemList)
            
            return(itemList, keyList)
                 
        def CF(itemList, keyList):
            """Change font given item"""
            # initialize fontprop object
            fontprop = fm.FontProperties(style='normal', weight='normal',
                                         stretch = 'normal')
            if os.path.isfile(fontname): # check if font is a file
                fontprop.set_file(fontname)
            else:# check if the name of font is available in the system
                if not any([fontname.lower() in a.lower() for a in
                        fm.findSystemFonts(fontpaths=None, fontext='ttf')]):
                     print('Cannot find specified font: %s' %(fontname))
                fontprop.set_family(fontname) # set font name
            # set font for each object
            for n, item in enumerate(itemList):
                if isinstance(fontsize, dict):
                    if keyList[n] in fontsize.keys():
                        fontprop.set_size(fontsize[keyList[n]])
                    else:
                        pass
                        # print('Warning font property {} not in specified fontsize. Font is kept at defualt.'.format(keyList[n]))
                elif n <1: # set the properties only once
                    fontprop.set_size(fontsize)
                item.set_fontproperties(fontprop) # change font for all items
            
        def CF_ax(ax): # combine CF and get_ax_items
            if not ax: # true when empty or None
                return # skip axis font change
            itemList, keyList = get_ax_items(ax)
            CF(itemList, keyList)
            
        def CF_fig(fig): # combine CF and get_fig_items
            if not fig: # true when empty or None
                return # skip figure font change
            itemsList, keyList = get_fig_items(fig)
            CF(itemsList, keyList)
        
        # vecotirze the closure
        CF_ax_vec = np.frompyfunc(CF_ax, 1,1)
        CF_fig_vec = np.frompyfunc(CF_fig, 1,1)
        
        # Do the actual font change
        CF_ax_vec(ax)
        CF_fig_vec(fig)

        
def AdjustAxs(otypes=[np.ndarray], excluded=None):
    """Used as a decorator to set the axis properties"""
    def wrap(func):
        # vectorize the func so that it can be applied to single axis or
        # multiple axes
        func_vec = np.vectorize(func, otypes=otypes, excluded=excluded)
        def wrapper(ax, *args, **kwargs):
            res = func_vec(ax, *args, **kwargs)
            return(res)
        return(wrapper)
    return(wrap)
    
def SetAxisOrigin(ax, xcenter='origin', ycenter='origin', xspine='bottom', yspine='left'):
    """Set the origin of the axis"""
    if xcenter == 'origin':
        xtick = ax.get_xticks()
        if max(xtick)<0:
            xcenter = max(xtick)
        elif min(xtick)>0:
            xcenter = min(xtick)
        else:
            xcenter = 0
            
    if ycenter == 'origin':
        ytick = ax.get_yticks()
        if max(ytick)<0:
            ycenter = max(ytick)
        elif min(ytick)>0:
            ycenter = min(ycenter)
        else:
            ycenter = 0
            
    xoffspine = 'top' if xspine == 'bottom' else 'bottom'     
    yoffspine = 'right' if yspine=='left' else 'left'
    
           
    ax.spines[xspine].set_position(('data', ycenter))
    ax.spines[yspine].set_position(('data', xcenter))
    ax.spines[xoffspine].set_visible(False)
    ax.spines[yoffspine].set_visible(False)
    ax.xaxis.set_ticks_position(xspine)
    ax.yaxis.set_ticks_position(yspine)
    ax.spines[xspine].set_capstyle('butt')
    ax.spines[yspine].set_capstyle('butt')
    
def xysize_pt2data(ax, s=None, dpi=None, scale=(1,1)):
    """ Determine dot size in data axis.
    scale: helps further increasing space between dots
    s: font size in points
    """
    figw, figh = ax.get_figure().get_size_inches() # figure width, height in inch
    dpi = float(ax.get_figure().get_dpi()) if dpi is None else float(dpi)
    w = (ax.get_position().xmax-ax.get_position().xmin)*figw # axis width in inch
    h = (ax.get_position().ymax-ax.get_position().ymin)*figh # axis height in inch
    xran = ax.get_xlim()[1]-ax.get_xlim()[0] # axis width in data
    yran = ax.get_ylim()[1]-ax.get_ylim()[0] # axis height in data
    
    xsize=np.sqrt(s)/dpi*xran/w*scale[0] # xscale * proportion of xwidth in data
    ysize=np.sqrt(s)/dpi*yran/h*scale[1] # yscale * proportion of yheight in data

    return xsize, ysize
    
def mm2pt(mm):
    return mm/10.0/2.54*72 # mm->cm-->inch-->pt
    
def xysize_data2pt(ax, p=None, dpi=None, scale=(1,1)):
    """Convert from data size to font size in points
    p: data size    
    """
    figw, figh = ax.get_figure().get_size_inches() # figure width, height in inch
    dpi = float(ax.get_figure().get_dpi()) if dpi is None else float(dpi)
    w = (ax.get_position().xmax-ax.get_position().xmin)*figw # axis width in inch
    h = (ax.get_position().ymax-ax.get_position().ymin)*figh # axis height in inch
    xran = ax.get_xlim()[1]-ax.get_xlim()[0] # axis width in data
    yran = ax.get_ylim()[1]-ax.get_ylim()[0] # axis height in data
    
    xsize = (p*w*dpi/xran*scale[0])**2
    ysize = (p*h*dpi/yran*scale[1])**2
    
    return xsize, ysize
    
def AdjustCategoricalXAxis(ax, pad=(0.5,0.5), categorytickon=False):
        """Additional settings for plots with categorical data"""
        # change the x lim on the last, most buttom subplot
        ax.set_xlim(ax.get_xticks()[0]-pad[0],ax.get_xlim()[-1])
        ax.set_xlim(ax.get_xlim()[0], ax.get_xticks()[-1]+pad[1])
        if not categorytickon:
            ax.tick_params(axis='x', bottom='off')
        
def AdjustCategoricalYAxis(ax, pad=(0.5,0.5), categorytickon=False):
    """Additional settings for plots with categorical data"""
    ax.set_ylim(ax.get_yticks()[0]-pad[0],ax.get_ylim()[-1])
    ax.set_ylim(ax.get_ylim()[0], ax.get_yticks()[-1]+pad[1])
    if not categorytickon:
        ax.tick_params(axis='y', left='off')

@AdjustAxs()
def SetDefaultAxis(ax):
    """Set default axis appearance"""
    ax.tick_params(axis='both',direction='out')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_capstyle('butt')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_capstyle('butt')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

@AdjustAxs()
def SetDefaultAxis3D(ax, elev=45, azim=60, dist=12):
    ax.tick_params(axis='both', direction='out')
    ax.view_init(elev=elev, azim=azim) # set perspective
    ax.dist = dist # use default axis distance 10
    if ax.azim > 0: # z axis will be on the left
        ax.zaxis.set_rotate_label(False) # prevent auto rotation
        a = ax.zaxis.label.get_rotation()
        ax.zaxis.label.set_rotation(90+a) # set custom rotation
        ax.invert_xaxis() # make sure (0,0) in front
        ax.invert_yaxis() # make sure (0,0) in front
    else:
        ax.invert_xaxis() # make sure (0,0) in front
    #ax.zaxis.label.set_color('red')
    #ax.yaxis._axinfo['label']['space_factor'] = 2.8

@AdjustAxs()
def equalAxLineWidth(ax, lineproperty ={'xaxis_tick': 0.70866144, 
                                        'yaxis_tick': 0.70866144, 
                                        'xaxis_spine': 0.70866144, 
                                        'yaxis_spine': 0.70866144}):
        ax.spines['left'].set_linewidth(bar_line_property['xaxis_spine'])
        ax.spines['right'].set_linewidth(bar_line_property['xaxis_spine'])
        ax.spines['bottom'].set_linewidth(bar_line_property['yaxis_spine'])
        ax.spines['top'].set_linewidth(bar_line_property['yaxis_spine'])
        ax.xaxis.set_tick_params(width=bar_line_property['xaxis_tick'])
        ax.yaxis.set_tick_params(width=bar_line_property['yaxis_tick'])
        
@AdjustAxs()
def setAxisLineStyle(ax, lineproperty={'xaxis_tick_capstyle':'projecting',
                                       'xaxis_tick_joinstyle':'miter',
                                       'yaxis_tick_capstyle':'projecting',
                                       'yaxis_tick_joinstyle':'miter',
                                       'xaxis_spine_capstyle':'projecting',
                                       'xaxis_spine_joinstyle':'miter',
                                       'yaxis_spine_capstyle':'projecting',
                                       'yaxis_spine_joinstyle':'miter',
                                       }):
    # Ticks
    for i in ax.xaxis.get_ticklines(): 
        i._marker._capstyle = lineproperty['xaxis_tick_capstyle']
        i._marker._joinstyle = lineproperty['xaxis_tick_joinstyle']
    
    for i in ax.yaxis.get_ticklines():
        i._marker._capstyle = lineproperty['yaxis_tick_capstyle']
        i._marker._joinstyle = lineproperty['yaxis_tick_joinstyle']
        
    # Spines
    ax.spines['left']._capstyle = lineproperty['yaxis_spine_capstyle']
    ax.spines['left']._joinstyle = lineproperty['yaxis_spine_joinstyle']
    ax.spines['right']._capstyle = lineproperty['yaxis_spine_capstyle']
    ax.spines['right']._joinstyle = lineproperty['yaxis_spine_joinstyle']
    ax.spines['top']._capstyle = lineproperty['xaxis_spine_capstyle']
    ax.spines['top']._joinstyle = lineproperty['xaxis_spine_joinstyle']
    ax.spines['bottom']._capstyle = lineproperty['xaxis_spine_capstyle']
    ax.spines['bottom']._joinstyle = lineproperty['xaxis_spine_joinstyle']

@AdjustAxs()
def defaultAxisStyle(ax):
    equalAxLineWidth(ax)
    setAxisLineStyle(ax)

    
def add_subplot_axes(ax,rect,axisbg='w'):
    """Adding subplot within a plot"""
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

    

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
          ExportOpts.compatibleGradientPrinting = true;
          ExportOpts.includeDocumentThumbnails = true;
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
               illustrator_path="D:/Edward/Software/Adobe Illustrator CS6/Support Files/Contents/Windows/Illustrator.exe",\
               jsx_file_str = jsx_file_str_AI_CS6, DEBUG=False):
    """Use Adobe Illustrator to convert svg to eps"""
    # Change the strings
    jsx_file_str = jsx_file_str.replace('{format_source_file}', source_file)
    jsx_file_str = jsx_file_str.replace('{format_target_file}', target_file).replace('\\','/')
    tmp_f = os.path.abspath(os.path.join(os.path.dirname(target_file), "tmp.jsx"))
    f = open(tmp_f, 'w')
    f.write(jsx_file_str)
    f.close()

    # Remove previous target file if already existed
    if os.path.isfile(target_file):
        os.remove(target_file)

    # subprocess.check_call([illustrator_path, '-run', tmp_f])
    cmd = " ".join(['"'+illustrator_path+'"', '-run', '"'+tmp_f+'"'])
    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # print(pro.stdout)
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

def save_svg2eps(fig, savepath):
    if '.eps' in savepath:
        savepath = savepath.replace('.eps', '.svg')
    fig.savefig(savepath, bbox_inches='tight', dpi=300, transparent=True) # save as svg first
    svg2eps_ai(savepath, savepath.replace('.svg', '.eps'))
    os.remove(savepath)
    
    
def plot_ci_manual(xdata, ydata, x_plot, y_plot, popt, alpha=0.95, ax=None, color="#b9cfe7", edgecolor="", *args, **kwargs):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1]: M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()
        
    DF = len(xdata) - len(popt)
    y_model = np.polyval(popt, xdata)
    resid = ydata - y_model
    chi2 = np.sum((resid/y_model)**2)
    # chi2_red = chi2/DF
    s_err = np.sqrt(np.sum(resid**2)/DF)
    
    t = stats.t.ppf(alpha, DF)
    ci = t*s_err*np.sqrt(1/len(xdata) + (x_plot-np.mean(xdata))**2/np.sum((xdata-np.mean(xdata))**2))
    ax.fill_between(x_plot, y_plot+ci, y_plot-ci, color=color, edgecolor=edgecolor, *args, **kwargs)

    return ax



