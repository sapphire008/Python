# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:46:03 2017

@author: Edward
"""

import cloudconvert
api = cloudconvert.Api('5PGyLT7eAn0yLbnBU3G-7j1JLFWTfcnFUk6x7k_lhuwzioGwqO7bVQ-lJNunsDkrr9fL1JDdjdVog6iDZ31yIw')
process = api.convert({"input": "upload",
                       "file": open('R:/temp.svg', 'rb'),
                       "inputformat": "svg",
                       "outputformat": "eps",
                       })

process.wait()
process.download()


