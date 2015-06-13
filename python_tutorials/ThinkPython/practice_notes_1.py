# Python 3.3.0 Practice Notes
# Day 1: November 23, 2012

# print()
#default delimiter is \n, which prints at a new line every line of print()
print('Hello, world!',"I am okay");#use single or double quotes are both fine

#len()
len("asdffg");#returns the length of the string

# Converting between letter and integer (ASCII)
ord('a'); #--> integer
chr(97);  #--> unicode character

# Concatenation
first = 'throat';
second = ' warbler';
print(first + second);
#another example of concatenation
this_word = 'Spam';
print(this_word*3);# 'this_word', then, will be repeated 3 times in one string

# Difference in Division between Python2 and Python3
#In Python 2, / is the floor division,
#whereas in Python 3, // is the floor division. This means, even if one of the number is float
#if we call // in division operation, it is going to perform a floor division first,
#Then convert the result to a float.
#In Python 2, to use float division, we must convert one of the number into floats
#whereas in Python 3, / is the float division

# Checking the type of a variable / object
type(32); #--><type 'int'>
type ('32'); #--><type 'str'>

# Type Conversion
int('32'); #--> 32 from type str to type int
int(3.99999); #--> 3
int(-2.3333); #--> 2
float(2); #-->2.0, from type int to type float
float('23.424'); # 23.424, from type str to type float
str(32.32); #-->'32.32', from type float to type str

# Math Modules and associated funtions
import math;#import math modules
print(math.pi);#returns constant pi
print(math.e);#returns natural number e
print(math.log(3,4));#returns log base 4 of 3
print(math.log10(20.3));#returns log base 10 of 20.3
print(math.log2(23));#returns log base 2 of 23, more accurate than using log(x,base)
print(math.exp(3));#returns e to the 3rd power
print(math.pow(2,3));#returns 2 raised to the 3rd power
print(math.sqrt(3));#returns square root of 3
#other functions
#math.sin, math.cos, math.tan,
#math.atan2 (returns value in radians between -pi and pi)
#math.degrees(x), math.radians(x)
#For complex number, "import cmath" instead of "import math"
#use cmath as the name of the module to call out these functions
#We may also do
from math import * #import all functions from the math module
pi #we now can use the functions from math directly, without typing math. every time

# Functions
math_eq1=1+1;
math_eq2=2+1;
math_eq3=math.pi;
def let_it_all_out(a,b,c):   #don't forget the colon after the parenthesis (which is for argument inputs)!
    print("Okay, let's do some math");
    print(a);
    print(b);
    print(c);
    print("Good Job!");
                        #an empty line to signal the end of the function
#now, call the function
let_it_all_out(math_eq1,math_eq2,math_eq3);

#This concludes today's study.