# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 22:19:17 2015

Returns additional custom color, line, and, marker styles / cycles

@author: Edward
"""

from collections import OrderedDict

def rgb2hex(c):
    """Convert from RGB triplets to hex string"""
    return('#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]))
    
def hex2rgb(v):
    """Convert a hex string to rgb tuple triplet"""
    v = v.lstrip('#')
    lv = len(v)
    return(tuple(int(v[i:i+lv/3], 16) for i in range(0, lv, lv/3)))
    
def rgbdecimal2int(c):
    """ scale RGB value from [0, 1] to [0, 255]"""
    return((int(c[0]*255.999), int(c[1]*255.999), int(c[2]*255.999) ))
    
def rgbint2decimal(c):
    """scale RGB value from [0, 255] to [0, 1]"""
    return(c[0]/255.0, c[1]/255.0, c[2]/255.0)
    
def ColorBrewer(cname='PuBuGn'):
    """http://colorbrewer2.org/, used by web designers and R's ggplot"""
    return({
    'BuGn' : OrderedDict([(0, (247,252,253)),(1, (229,245,249)),(2, (204,236,230)),(3, (153,216,201)),(4, (102,194,164)),(5, (65,174,118)),(6, (35,139,69)),(7, (0,109,44)),(8, (0,68,27))]),
    'BuPu' : OrderedDict([(0, (247,252,253)),(1, (224,236,244)),(2, (191,211,230)),(3, (158,188,218)),(4, (140,150,198)),(5, (140,107,177)),(6, (136,65,157)),(7, (129,15,124)),(8, (77,0,75))]),
    'GnBu' : OrderedDict([(0, (247,252,240)),(1, (224,243,219)),(2, (204,235,197)),(3, (168,221,181)),(4, (123,204,196)),(5, (78,179,211)),(6, (43,140,190)),(7, (8,104,172)),(8, (8,64,129))]),
    'OrRd' : OrderedDict([(0, (255,247,236)),(1, (254,232,200)),(2, (253,212,158)),(3, (253,187,132)),(4, (252,141,89)),(5, (239,101,72)),(6, (215,48,31)),(7, (179,0,0)),(8, (127,0,0))]),
    'PuBu' : OrderedDict([(0, (255,247,251)),(1, (236,231,242)),(2, (208,209,230)),(3, (166,189,219)),(4, (116,169,207)),(5, (54,144,192)),(6, (5,112,176)),(7, (4,90,141)),(8, (2,56,88))]),
    'PuBuGn' : OrderedDict([(0, (255,247,251)),(1, (236,226,240)),(2, (208,209,230)),(3, (166,189,219)),(4, (103,169,207)),(5, (54,144,192)),(6, (2,129,138)),(7, (1,108,89)),(8, (1,70,54))]),
    'PuRd' : OrderedDict([(0, (247,244,249)),(1, (231,225,239)),(2, (212,185,218)),(3, (201,148,199)),(4, (223,101,176)),(5, (231,41,138)),(6, (206,18,86)),(7, (152,0,67)),(8, (103,0,31))]),
    'RdPu' : OrderedDict([(0, (255,247,243)),(1, (253,224,221)),(2, (252,197,192)),(3, (250,159,181)),(4, (247,104,161)),(5, (221,52,151)),(6, (174,1,126)),(7, (122,1,119)),(8, (73,0,106))]),
    'YlGn' : OrderedDict([(0, (255,255,229)),(1, (247,252,185)),(2, (217,240,163)),(3, (173,221,142)),(4, (120,198,121)),(5, (65,171,93)),(6, (35,132,67)),(7, (0,104,55)),(8, (0,69,41))]),
    'YlGnBu' : OrderedDict([(0, (255,255,217)),(1, (237,248,177)),(2, (199,233,180)),(3, (127,205,187)),(4, (65,182,196)),(5, (29,145,192)),(6, (34,94,168)),(7, (37,52,148)),(8, (8,29,88))]),
    'YlOrBr' : OrderedDict([(0, (255,255,229)),(1, (255,247,188)),(2, (254,227,145)),(3, (254,196,79)),(4, (254,153,41)),(5, (236,112,20)),(6, (204,76,2)),(7, (153,52,4)),(8, (102,37,6))]),
    'YlOrRd' : OrderedDict([(0, (255,255,204)),(1, (255,237,160)),(2, (254,217,118)),(3, (254,178,76)),(4, (253,141,60)),(5, (252,78,42)),(6, (227,26,28)),(7, (189,0,38)),(8, (128,0,38))]),
    'Blues' : OrderedDict([(0, (247,251,255)),(1, (222,235,247)),(2, (198,219,239)),(3, (158,202,225)),(4, (107,174,214)),(5, (66,146,198)),(6, (33,113,181)),(7, (8,81,156)),(8, (8,48,107))]),
    'Greens' : OrderedDict([(0, (247,252,245)),(1, (229,245,224)),(2, (199,233,192)),(3, (161,217,155)),(4, (116,196,118)),(5, (65,171,93)),(6, (35,139,69)),(7, (0,109,44)),(8, (0,68,27))]),
    'Greys' : OrderedDict([(0, (255,255,255)),(1, (240,240,240)),(2, (217,217,217)),(3, (189,189,189)),(4, (150,150,150)),(5, (115,115,115)),(6, (82,82,82)),(7, (37,37,37)),(8, (0,0,0))]),
    'Oranges' : OrderedDict([(0, (255,245,235)),(1, (254,230,206)),(2, (253,208,162)),(3, (253,174,107)),(4, (253,141,60)),(5, (241,105,19)),(6, (217,72,1)),(7, (166,54,3)),(8, (127,39,4))]),
    'Purples' : OrderedDict([(0, (252,251,253)),(1, (239,237,245)),(2, (218,218,235)),(3, (188,189,220)),(4, (158,154,200)),(5, (128,125,186)),(6, (106,81,163)),(7, (84,39,143)),(8, (63,0,125))]),
    'Reds' : OrderedDict([(0, (255,245,240)),(1, (254,224,210)),(2, (252,187,161)),(3, (252,146,114)),(4, (251,106,74)),(5, (239,59,44)),(6, (203,24,29)),(7, (165,15,21)),(8, (103,0,13))])
    }.get(cname))
    
def Tableau(cname='tableau10'):
    """tableau color
    tableau20 color naming from: 
    https://gist.github.com/Nepomuk/859fef81a912a9fe425e
    """
    return({
    'tableau20':OrderedDict([('steelblue',(31, 119, 180)),('lightsteelblue',(174, 199, 232)),('darkorange',(255, 127, 14)), ('peachpuff',(255, 187, 120)), ('green',(44, 160, 44)), ('lightgreen',(152, 223, 138)),('crimson',(214, 39, 40)), ('lightcoral',(255, 152, 150)),('mediumpurple',(148, 103, 189)), ('thistle',(197, 176, 213)), ('saddlebrown',(140, 86, 75)),('rosybrown',(196, 156, 148)),('orhchid',(227, 119, 194)),('lightpink',(247, 182, 210)),('gray',(127, 127, 127)), ('lightgray',(199, 199, 199)),('olive',(188, 189, 34)),('palegoldenrod',(219, 219, 141)), ('mediumtorquoise',(23, 190, 207)),('paleturqoise',(158, 218, 229))]),
    'tableau10':OrderedDict([('steelblue',(31,119,180)),('darkorange',(255,127,14)),('green',(44,160,44)),('crimson',(214,39,40)),('mediumpurple',(148,103,189)),('saddlebrown',(140,86,75)),('orhchid',(227,119,194)),('gray',(127,127,127)),('olive',(188,189,340)),('mediumtorquoise',(23,190,207))]),
    'tableau10light':OrderedDict([('lightsteelblue',(174, 199, 232)),('peachpuff',(255, 187, 120)),('lightgreen',(152, 223, 138)),('lightcoral',(255, 152, 150)),('thistle',(197, 176, 213)),('rosybrown',(196, 156, 148)),('lightpink',(247, 182, 210)),('lightgray',(199, 199, 199)),('palegoldenrod',(219, 219, 141)),('paleturqoise',(158, 218, 229))]),
    'tableau10medium':OrderedDict([('cerulean',(114,158,206)),('orange',(255,158,74)),('younggreen',(103,191,92)),('red',(237,102,93)),('violet',(173,139,201)),('cocoa',(168,120,110)),('pink',(237,151,202)),('silver',(162,162,162)),('witheredyellow',(205,204,93)),('aqua',(109,204,218))]),
    'tableau10blind':OrderedDict([('deepskyblue4',(0, 107, 164)),('darkorange1',(255, 128,  14)),('darkgray',(171, 171, 171)),('dimgray',( 89,  89,  89)),('skyblue3',( 95, 158, 209)),('chocolate3',(200,  82,   0)),('gray',(137, 137, 137)),('slategray1',(163, 200, 236)),('sandybrown',(255, 188, 121)),('lightgray',(207, 207, 207))]),
    'tableaugray5': OrderedDict([('gray1',(207,207,207)),('gray2',(165,172,175)),('gray3',(143,135,130)),('gray4',(96,99,106)),('gray5',(65,68,81))])
    }.get(cname))

def Colors(cname='tableau10', Hex=True, returnOnly='code', reverseOrder=False):
    """A list of colors in RGB
    cname: color name. Default 'tableau20'
    returnOnly: ['code'|'name'], return only RGB color code or color name
            as a list
    Invert: (True / False) inverse the color order, default ordering from 
            light to dark hues
    """
    if cname in ['tableau20', 'tableau10', 'tableau10light', 'tableau10medium', 'tableau10blind', 'tableaugray5']:
        colors = Tableau(cname)
    elif cname in ['BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'Blues', 'Greens', 'Greys', 'Oranges', 'Purples', 'Reds']:    
        # http://colorbrewer2.org/, used by web designers and R's ggplot
        colors = ColorBrewer(cname)
    else:
        # Other custom colors
        colors = {
                  'matlab':OrderedDict([('black',(0,0,0)),('red',(255,0,0)),('blue',(0,0,255)), ('orange',(255,165,0)),('green',(0,127,0)), ('cyan', (0, 191,191)),('magenta', (191, 0, 191))])
                  }.get(cname,Tableau('tableau20'))
    # invert the color order
    if reverseOrder:
        colors = OrderedDict(list(reversed(list(colors.items()))))
    # convert to html hex strings
    if Hex:
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


def Lines(lname='continuous', returnOnly='code', invert=False):
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
    