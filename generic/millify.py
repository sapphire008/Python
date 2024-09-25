"""
Copying this from https://github.com/azaitsev/millify/blob/master/millify/__init__.py
so that there is one dependency less to maintain. 

Modifications: 
* Allow the millify function to handle dollar signs and other potential prefixes.
* Allow users to prettify and millify given a unit, e.g. 12,345k or 12.345M


Original license reproduced below:

MIT License

Copyright (c) 2018 Alex Zaitsev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import math
import re
from decimal import Decimal
from typing import Union, Text
from ipdb import set_trace

__author__ = (
    "Alexander Zaitsev (azaitsev@gmail.com), Edward Cui (cui23327@gmail.com)"
)
__copyright__ = "Copyright 2024, azaitsev@gmail.com, cui23327@gmail.com"
__license__ = "MIT"
__version__ = "0.1.2"


def remove_exponent(d):
    """Remove exponent."""
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()


def _prettify(amount, separator=","):
    """Separate with predefined separator.
    prettify(1234)
    # '1,234'

    prettify('1234') # same for strings
    # '1,234'

    prettify(1234, '`')
    # '1`234'
    """
    orig = str(amount)
    new = re.sub(
        "^(-?\d+)(\d{3})", "\g<1>{0}\g<2>".format(separator), str(amount)
    )
    if orig == new:
        return new
    else:
        return _prettify(new)
    
def _get_mill_index(n, millnames):
    millidx = max(
            0,
            min(
                len(millnames) - 1,
                int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
            ),
        )
    basis = 10 ** (3 * millidx)
    return millidx, basis


def millify(
    n: Union[int, float],
    unit: Text = "auto",
    precision: int = 0,
    drop_nulls: bool = False,
    prefix: Text="",
    suffix: Text="",
    prettify: bool = True,
):
    """
    Humanize number.
    * n: input number (float, int)
    * unit: set a fixed unit to use rather than determining automatically, using a 
        value from the list ['', 'k', 'M', 'B', 'T', 'P', 'E', 'Z', 'Y'].
        Default is "auto".
    * precision: float-point precision. Default to 0
    * drop_nulls: Dropping ".00" in float and show as an integer.
        Default to False.
    * prefix: prefix units such as dollar signs $, which will be
        appended before the number but after the negative sign.
        Default to empty string ""
    * suffix: suffix of the unit. Default to empty string "".
    * prettify: adding comma separation for large numbers after millify. 
        Default is True.

    Examples:

    millify(1234)
    # '1k'

    millify('1234') # same for strings
    # '1k'

    millify(12345678)
    # '12M'

    millify(12345678, precision=2)
    # '12.35M'
    
    millify(10000, precision=2, drop_nulls=False)
    # '10.00k'

    # hide nulls ".00" in decimals
    millify(10000, precision=2, drop_nulls=True)
    # '10k'

    # Negative number with prefix
    millify(-510_638.46, precision=0,  unit="M", prefix="$", suffix="B")
    '-$1MB'
    
    millify(5_100_638.46, precision=6,  unit="k", prefix="$", prettify=True)
    '$5,100.63846k'
    
    # "up-to": Already reached Million
    millify(5_100_638.46, precision=2, unit="up-to:M", prefix="$", prettify=True)
    '$5.10M'
    
    # "up-to": Has not reached Million, use "auto" mode
    millify(500_638.46, precision=2,  unit="up-to:M", prefix="$", prettify=True)
    '$500.64k'
    """
    millnames = ["", "k", "M", "B", "T", "P", "E", "Z", "Y"]
    sign = "-" if n < 0 else ""
    n = float(abs(n))
    if unit == "auto":
        millidx, basis = _get_mill_index(n, millnames)
    else:
        if "up-to:" in unit:
            # Only starts to millify if reaching that magnitude of the unit
            unit = unit.replace("up-to:", "")
            upto = True
        else:
            upto = False
        
        try:
            millidx = millnames.index(unit)
        except:
            raise(ValueError(f"Invalid unit {unit}. Must be one of {millnames}"))
        
        # Try to see if the number has reached this magnitude
        basis = 10 ** (3 * millidx)
        
        if n < basis and upto:
            # not reaching the magnitude, getting it from auto
            millidx, basis = _get_mill_index(n, millnames)
            
    result = "{:.{precision}f}".format(
        n / basis, precision=precision
    )
    if drop_nulls:
        result = remove_exponent(Decimal(result))
    if prettify:
        result = _prettify(result)
    
    # Assemble the final output
    dx = millnames[millidx]
    return f"{sign}{prefix}{result}{dx}{suffix}"
