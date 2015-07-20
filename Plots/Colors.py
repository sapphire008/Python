# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 22:19:17 2015

Returns additional custom color palette

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

def rgb2cmyk(rgb,cmyk_scale = 1.0):
    if tuple(rgb)==(0.,0.,0.):
        # black
        return(0., 0., 0., cmyk_scale)

    # rgb [0,255] -> cmy [0,1]
    c,m,y = [1.0-a/255.0 for a in rgb]
    # extract out k [0,1]
    k = min(c, m, y)
    c,m,y = [(a - k)/(1.0-k) for a in (c,m,y)]
    # rescale to the range [0,cmyk_scale]
    return(c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale)

def rgbdecimal2int(c):
    """ scale RGB value from [0, 1] to [0, 255]"""
    return((int(c[0]*255.999), int(c[1]*255.999), int(c[2]*255.999) ))

def rgbint2decimal(c):
    """scale RGB value from [0, 255] to [0, 1]"""
    return(c[0]/255.0, c[1]/255.0, c[2]/255.0)

def printCSS(c):
    """Print CSS array of colors for copy and paste into CSS script"""

    if type(c) == 'collections.OrderedDict':
        c = c.values()
    clist = " ".join([".Set .q%d-%d{fill.:rgb(%d,%d,%d)}" %((n,len(c)-1)+x) for n,x in enumerate(c)])
    print(clist) #.Set3 .q0-8{fill:rgb(141,211,199)} ...
    return(clist)

def printJS(c):
    """print JavaScript array of colors for copy and paste into JS script"""
    if type(c) == 'collections.OrderedDict':
        c = c.values()
    clist = ",".join(["'rgb(%d,%d,%d)'"%(x) for x in c])
    clist = "["+clist+"]"
    print(clist) # ['rgb(255,245,240)',...]
    return(clist)

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
    'Reds' : OrderedDict([(0, (255,245,240)),(1, (254,224,210)),(2, (252,187,161)),(3, (252,146,114)),(4, (251,106,74)),(5, (239,59,44)),(6, (203,24,29)),(7, (165,15,21)),(8, (103,0,13))]),
    'BrBG' : OrderedDict([(0, (84,48,5)),(1, (140,81,10)),(2, (191,129,45)),(3, (223,194,125)),(4, (246,232,195)),(5, (245,245,245)),(6, (199,234,229)),(7, (128,205,193)),(8, (53,151,143)),(9, (1,102,94)),(10, (0,60,48))]),
    'PiYG' : OrderedDict([(0, (142,1,82)),(1, (197,27,125)),(2, (222,119,174)),(3, (241,182,218)),(4, (253,224,239)),(5, (247,247,247)),(6, (230,245,208)),(7, (184,225,134)),(8, (127,188,65)),(9, (77,146,33)),(10, (39,100,25))]),
    'PRGn' : OrderedDict([(0, (64,0,75)),(1, (118,42,131)),(2, (153,112,171)),(3, (194,165,207)),(4, (231,212,232)),(5, (247,247,247)),(6, (217,240,211)),(7, (166,219,160)),(8, (90,174,97)),(9, (27,120,55)),(10, (0,68,27))]),
    'PuOr' : OrderedDict([(0, (127,59,8)),(1, (179,88,6)),(2, (224,130,20)),(3, (253,184,99)),(4, (254,224,182)),(5, (247,247,247)),(6, (216,218,235)),(7, (178,171,210)),(8, (128,115,172)),(9, (84,39,136)),(10, (45,0,75))]),
    'RdBu' : OrderedDict([(0, (103,0,31)),(1, (178,24,43)),(2, (214,96,77)),(3, (244,165,130)),(4, (253,219,199)),(5, (247,247,247)),(6, (209,229,240)),(7, (146,197,222)),(8, (67,147,195)),(9, (33,102,172)),(10, (5,48,97))]),
    'RdGy' : OrderedDict([(0, (103,0,31)),(1, (178,24,43)),(2, (214,96,77)),(3, (244,165,130)),(4, (253,219,199)),(5, (255,255,255)),(6, (224,224,224)),(7, (186,186,186)),(8, (135,135,135)),(9, (77,77,77)),(10, (26,26,26))]),
    'RdYlBu' : OrderedDict([(0, (165,0,38)),(1, (215,48,39)),(2, (244,109,67)),(3, (253,174,97)),(4, (254,224,144)),(5, (255,255,191)),(6, (224,243,248)),(7, (171,217,233)),(8, (116,173,209)),(9, (69,117,180)),(10, (49,54,149))]),
    'RdYlGn' : OrderedDict([(0, (165,0,38)),(1, (215,48,39)),(2, (244,109,67)),(3, (253,174,97)),(4, (254,224,139)),(5, (255,255,191)),(6, (217,239,139)),(7, (166,217,106)),(8, (102,189,99)),(9, (26,152,80)),(10, (0,104,55))]),
    'Spectral' : OrderedDict([(0, (158,1,66)),(1, (213,62,79)),(2, (244,109,67)),(3, (253,174,97)),(4, (254,224,139)),(5, (255,255,191)),(6, (230,245,152)),(7, (171,221,164)),(8, (102,194,165)),(9, (50,136,189)),(10, (94,79,162))]),
    'Paired' : OrderedDict([(0, (166,206,227)),(1, (31,120,180)),(2, (178,223,138)),(3, (51,160,44)),(4, (251,154,153)),(5, (227,26,28)),(6, (253,191,111)),(7, (255,127,0)),(8, (202,178,214)),(9, (106,61,154)),(10, (255,255,153)),(11, (177,89,40))]),
    'Pastel1' : OrderedDict([(0, (251,180,174)),(1, (179,205,227)),(2, (204,235,197)),(3, (222,203,228)),(4, (254,217,166)),(5, (255,255,204)),(6, (229,216,189)),(7, (253,218,236)),(8, (242,242,242))]),
    'Pastel2' : OrderedDict([(0, (179,226,205)),(1, (253,205,172)),(2, (203,213,232)),(3, (244,202,228)),(4, (230,245,201)),(5, (255,242,174)),(6, (241,226,204)),(7, (204,204,204))]),
    'Set1' : OrderedDict([(0, (228,26,28)),(1, (55,126,184)),(2, (77,175,74)),(3, (152,78,163)),(4, (255,127,0)),(5, (255,255,51)),(6, (166,86,40)),(7, (247,129,191)),(8, (153,153,153))]),
    'Set2' : OrderedDict([(0, (102,194,165)),(1, (252,141,98)),(2, (141,160,203)),(3, (231,138,195)),(4, (166,216,84)),(5, (255,217,47)),(6, (229,196,148)),(7, (179,179,179))]),
    'Set3' : OrderedDict([(0, (141,211,199)),(1, (255,255,179)),(2, (190,186,218)),(3, (251,128,114)),(4, (128,177,211)),(5, (253,180,98)),(6, (179,222,105)),(7, (252,205,229)),(8, (217,217,217)),(9, (188,128,189)),(10, (204,235,197)),(11, (255,237,111))]),
    'Accent' : OrderedDict([(0, (127,201,127)),(1, (190,174,212)),(2, (253,192,134)),(3, (255,255,153)),(4, (56,108,176)),(5, (240,2,127)),(6, (191,91,23)),(7, (102,102,102))]),
    'Dark2' : OrderedDict([(0, (27,158,119)),(1, (217,95,2)),(2, (117,112,179)),(3, (231,41,138)),(4, (102,166,30)),(5, (230,171,2)),(6, (166,118,29)),(7, (102,102,102))])
    }.get(cname))

def Tableau(cname='tableau10'):
    """tableau color
    tableau20 color naming from:
    https://gist.github.com/Nepomuk/859fef81a912a9fe425e
    """
    return({
    'tableau20':OrderedDict([('steelblue',(31, 119, 180)),('lightsteelblue',(174, 199, 232)),('darkorange',(255, 127, 14)), ('peachpuff',(255, 187, 120)), ('green',(44, 160, 44)), ('lightgreen',(152, 223, 138)),('crimson',(214, 39, 40)), ('lightcoral',(255, 152, 150)),('mediumpurple',(148, 103, 189)), ('thistle',(197, 176, 213)), ('saddlebrown',(140, 86, 75)),('rosybrown',(196, 156, 148)),('orhchid',(227, 119, 194)),('lightpink',(247, 182, 210)),('gray',(127, 127, 127)), ('lightgray',(199, 199, 199)),('olive',(188, 189, 34)),('palegoldenrod',(219, 219, 141)), ('mediumtorquoise',(23, 190, 207)),('paleturqoise',(158, 218, 229))]),
    'tableau10':OrderedDict([('steelblue',(31,119,180)),('darkorange',(255,127,14)),('green',(44,160,44)),('crimson',(214,39,40)),('mediumpurple',(148,103,189)),('saddlebrown',(140,86,75)),('orhchid',(227,119,194)),('gray',(127,127,127)),('olive',(188,189,34)),('mediumtorquoise',(23,190,207))]),
    'tableau10light':OrderedDict([('lightsteelblue',(174, 199, 232)),('peachpuff',(255, 187, 120)),('lightgreen',(152, 223, 138)),('lightcoral',(255, 152, 150)),('thistle',(197, 176, 213)),('rosybrown',(196, 156, 148)),('lightpink',(247, 182, 210)),('lightgray',(199, 199, 199)),('palegoldenrod',(219, 219, 141)),('paleturqoise',(158, 218, 229))]),
    'tableau10medium':OrderedDict([('cerulean',(114,158,206)),('orange',(255,158,74)),('younggreen',(103,191,92)),('red',(237,102,93)),('violet',(173,139,201)),('cocoa',(168,120,110)),('pink',(237,151,202)),('silver',(162,162,162)),('witheredyellow',(205,204,93)),('aqua',(109,204,218))]),
    'tableau10blind':OrderedDict([('deepskyblue4',(0, 107, 164)),('darkorange1',(255, 128,  14)),('darkgray',(171, 171, 171)),('dimgray',( 89,  89,  89)),('skyblue3',( 95, 158, 209)),('chocolate3',(200,  82,   0)),('gray',(137, 137, 137)),('slategray1',(163, 200, 236)),('sandybrown',(255, 188, 121)),('lightgray',(207, 207, 207))]),
    'tableaugray5': OrderedDict([('gray1',(207,207,207)),('gray2',(165,172,175)),('gray3',(143,135,130)),('gray4',(96,99,106)),('gray5',(65,68,81))])
    }.get(cname))

def MATLAB(cname='matlabnew'):
    """MATLAB color scheme"""
    return({
    'matlabnew': OrderedDict([('blue',(0, 114, 189)),('orange',(217, 83, 25)),('yellow',(237, 177, 31)), ('purple',(126, 47, 142)),('green',(119, 172, 48)),('skyblue',(77, 190, 238)),('crimson',(162, 19, 47))]),
    'matlabold': OrderedDict([('black',(0,0,0)),('red',(255,0,0)),('blue',(0,0,255)), ('orange',(255,165,0)),('green',(0,127,0)), ('cyan', (0, 191,191)),('magenta', (191, 0, 191))])
    }).get(cname)


class ColorPalette(object):
    """Color Pallete utility"""
    def __init__(self, palette=None):
        """Initialization
        """
        self.palette = palette
        if self.palette is not None:
            self.colors = self.get_palette(palette)

    def get_palette(self,palette='tableau10',Hex=True, returnOnly='code',
                       reverseOrder=False):
        """Instance method for _get_palette"""
        self.colors = self._get_palette(palette,Hex,returnOnly,reverseOrder)

    @classmethod
    def _get_palette(cls, palette='tableau10', returnType='hex', returnOnly='code',
                    reverseOrder=False):
        """A list of colors in RGB
        cname: color name. Default 'tableau20'
        returnOnly: ['code'|'name'], return only RGB color code or color name
                as a list
        Invert: (True / False) inverse the color order, default ordering from
                light to dark hues
        """
        if palette in cls.list_palette('tableau'):
            colors = Tableau(palette)
        elif palette in cls.list_palette('colorbrewer'):
            # http://colorbrewer2.org/, used by web designers and R's ggplot
            colors = ColorBrewer(palette)
        elif palette in cls.list_palette('matlab'):
            colors = MATLAB(palette)
        else:# Other custom colors
            colors = Tableau('tableau10')
        if reverseOrder:
            colors = OrderedDict(list(reversed(list(colors.items()))))
            # convert to html hex strings
        colors = {
        'dec': OrderedDict([(k,rgbint2decimal(colors[k])) 
                                    for k in colors.keys()]),
        'rgb': colors,
        }.get(returnType, OrderedDict([(k, rgb2hex(colors[k]))
                                    for k in colors.keys()])) # ddefulat hex
        # Return
        return({'code': list(colors.values()),'name': list(colors.keys())
            }.get(returnOnly, colors))

    @classmethod
    def list_palette(cls, scheme='tableau'):
        return({
        'tableau': ['tableau20', 'tableau10', 'tableau10light', 'tableau10medium', 'tableau10blind', 'tableaugray5'],
        'colorbrewer': ['Spectral','Pastel2','RdPu','RdYlGn','PuOr','Greens','PRGn','Accent','OrRd','YlGnBu','RdYlBu','Paired','RdGy','PuBu','Set3','BrBG','Purples','Reds','YlOrRd','Pastel1','RdBu','GnBu','BuPu','Dark2','Greys','Oranges','BuGn','Set2','PiYG','YlOrBr','PuRd','Blues','PuBuGn','YlGn','Set1'],
        'matlab':['matlabnew', 'matlabold']
        }.get(scheme))

    @classmethod
    def list_scheme(cls):
        print('tableau; colorbrewer; matlab')
        
    def show_palette(self, palette='tableau'):
        """class instance of _show_all_palette"""
        self.fig, self.axs = self._show_all_palette(palette)
        
    @classmethod
    def _show_all_palette(cls,palette='tableau'):
        """
        Plot all palettes in a scheme
        """
        import matplotlib.pyplot as plt
        # get the list of palettes
        plist = cls.list_palette(palette)
        npalette = len(plist)
        # start the figure
        fig, axs = plt.subplots(nrows=npalette,ncols=1)
        for n,pname in enumerate(plist):
            # get the  color
            colors = cls._get_palette(pname,returnType='dec',returnOnly='code') 
            cname = cls._get_palette(pname,returnOnly='name') # get color name
            cls.palette_ax(axs[n], colors, pname, cname)
        plt.show()
        fig.tight_layout()
        return(fig, axs)

    @staticmethod
    def palette_ax(ax, colors, pname="", cname=[]):
        """Plot one palette"""
        from matplotlib.colors import ListedColormap
        import numpy as np
        colors = np.array([list(c) for c in colors])
        gradient = np.linspace(0, 1, len(colors))
        gradient = np.vstack((gradient, gradient))
        cmap = ListedColormap(colors)
        ax.imshow(gradient,aspect='auto',cmap=cmap, origin='lower',
                   interpolation='none')
        #plt.xticks(range(len(colors)),cname,rotation='vertical')
        ax.set_ylabel(pname)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', left='off',right='off',
                       top='off',bottom='off',
                       labelleft='off',labelbottom='off')

                
if __name__=="__main__":
    fig, ax = ColorPalette()._show_all_palette(palette='tableau')