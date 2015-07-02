# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 22:19:17 2015

Returns additional custom color, line, and, marker styles / cycles

@author: Edward
"""

from collections import OrderedDict

def rgb2hex(c):
    """Convert from RGB triplets to hex string"""
    return '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])
    
def hex2rgb(v):
    """Convert a hex string to rgb tuple triplet"""
    v = v.lstrip('#')
    lv = len(v)
    return tuple(int(v[i:i+lv/3], 16) for i in range(0, lv, lv/3))
    
def rgbdecimal2int(c):
    """ scale RGB value from [0, 1] to [0, 255]"""
    return((int(c[0]*255.999), int(c[1]*255.999), int(c[2]*255.999) ))
    
def rgbint2decimal(c):
    """scale RGB value from [0, 255] to [0, 1]"""
    return(c[0]/255.0, c[1]/255.0, c[2]/255.0)


def Colors(cname='tableau20_odd', returnAs='hex', returnOnly='code', invert=False):
    """A list of colors in RGB
    cname: color name. Default 'tableau20'
    returnOnly: ['code'|'name'], return only RGB color code or color name
            as a list
    Invert: (True / False) inverse the color order, default ordering from 
            light to dark hues
    """
    # tableau20 color naming from: 
    # https://gist.github.com/Nepomuk/859fef81a912a9fe425e
    tableau20 = OrderedDict([('steelblue',(31, 119, 180)),
    ('lightsteelblue',(174, 199, 232)),('darkorange',(255, 127, 14)), 
    ('peachpuff',(255, 187, 120)), ('green',(44, 160, 44)), 
    ('lightgreen',(152, 223, 138)),('crimson',(214, 39, 40)), 
    ('lightcoral',(255, 152, 150)),('mediumpurple',(148, 103, 189)), 
    ('thistle',(197, 176, 213)), ('saddlebrown',(140, 86, 75)),
    ('rosybrown',(196, 156, 148)),('orhchid',(227, 119, 194)),
    ('lightpink',(247, 182, 210)),('gray',(127, 127, 127)), 
    ('lightgray',(199, 199, 199)),('olive',(188, 189, 34)),
    ('palegoldenrod',(219, 219, 141)), ('mediumtorquoise',(23, 190, 207)),
    ('paleturqoise',(158, 218, 229))])
    # tableau 20 at odd index
    tableau20_odd = OrderedDict([(k, tableau20[k]) 
                                for k in tableau20.keys()[0::2]])
    
    # R's ggplot2: http://colorbrewer2.org/
    # From light to dark ordered blue shades
    bluegreenhue = OrderedDict([(1,(247,252,240)), (2,(224,243,219)), 
                                (3,(204,235,197)), (4,(168,221,181)),
                                (5,(123,204,196)), (6,(78,179,211)),
                                (7,(43,140,190)),  (8,(8,104,172)),
                                (9,(8,64,129))])
    # From light to dark ordered red shades
    redhue = OrderedDict([(1,(255,245,240)),  (2,(254,224,210)), 
                          (3,(252,187,161)), (4,(252,146,114)),
                          (5,(251,106,74)),  (6,(239,59,44)),
                          (7,(203,24,29)),   (8,(165,15,21)),
                          (9,(103,0,13))])
    # From light to dark ordered gray shades, excluding white and black
    grayhue = OrderedDict([(1,( 240,240,240)),  (2,(217,217,217)), 
                           (3,(189,189,189)), (4,(150,150,150)),
                           (5,(115,115,115)),  (6,(82,82,82))])
    # classic MATLAB color set
    matlab = OrderedDict([('black',(0,0,0)),('red',(255,0,0)), 
                          ('blue',(0,0,255)), ('orange',(255,165,0)),
                          ('green',(0,127,0)), ('cyan', (0, 191,191)),
                          ('magenta', (191, 0, 191))])
    # Get color
    colors = {'tableau20':tableau20, 'tableau20_odd':tableau20_odd, 
              'grayhue':grayhue, 'bluegreenhue':bluegreenhue, 
              'redhue':redhue, 'matlab':matlab
              }.get(cname,tableau20_odd)
    # invert the color order
    if invert:
        colors = OrderedDict(list(reversed(list(colors.items()))))
    if returnAs == 'hex':
        colors = OrderedDict([(k, rgb2hex(colors[k]))
                                for k in colors.keys()])
    # Return
    return({'code': list(colors.values()),
            'name': list(colors.keys())
            }.get(returnOnly, colors))
            

def Markers(mname='filled', returnOnly='code', invert=False):
    """
    A list of markers for scatter plot
    """
    #Based on guide 
    #http://www.labri.fr/perso/nrougier/teaching/matplotlib/#scatter-plots
    filled = OrderedDict([('circle','o'), ('square','s'), 
              ('thindiamond','d'), ('triangle','^'), ('star', '*'), 
              ('pentagon', 'p')])
    thin = OrderedDict([('point', '.'), ('plus','+'), ('cross','x'), 
                        ('snow',(5,2)), ('tripod', '4')])
    # Get marker
    markers = {'filled':filled, 'thin':thin}.get(mname, filled)
    # invert the marker order
    if invert:
        markers = OrderedDict(list(reversed(list(markers.items()))))
    # Return
    return({'code': list(markers.values()),
            'name': list(markers.keys())
            }.get(returnOnly, markers))


def Lines(self, lname='continuous', returnOnly='code', invert=False):
    """A list of markers for line plot"""
    continuous = OrderedDict([('solid', '-'), ('dashed','--'), 
                              ('dashdot','-.'),('dotted', ':')])
    # Get line
    lines = {'continuous':continuous}.get(lname, continuous)
    # invert the marker order
    if invert:
        lines = OrderedDict(list(reversed(list(lines.items()))))
    # Return
    return({'code': list(lines.values()),
            'name': list(lines.keys())
            }.get(returnOnly, lines))
    