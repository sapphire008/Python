# Python 3.3.0 Practice Notes
# Day 2: November 24, 2012
import math;

# Conditional statements
x=1;
y=2;
z=3;
if x<y and z<y:   #again, don't forget the colon
    print("Y is teh biggest!");
elif x<y or z<y:
    print("Let's do nothing!");
else:
    print("Okay, I am wrong");
                                    #again, additional empty line for ending this stub

# Recursion: a recursive function is a function that calls itself
def countdown(n):
    if n<=0:
        print("Balst Off!");
    else:
        print(n);
        countdown(n-1);

countdown(3)
#The function itself is like a while loop:
#As long as the else statement is executed (by calling itself),
#the loop continues until that the function no longer calls itself
#The output looks like this:
#>>>3
#>>>2
#>>>1
#>>>Blast Off!

# User Prompt
ask_question = input("Do you like Python? [Y/N]:\n");#asking user to type in something, str, int, float, etc...
if ask_question=="Y":
    print("Me, too");
else:
    ask_another_question = input("Why not?\n");
    print("Oh, okay, I see.");

#Note: % or some other symbol in the Python Shell prompts user input

# Non-void functions
abs(-3.323);#returns the absolute value of the input number

# Iteration
#for loop
for i in range(0,4,1):
    print("Hello, World!");

#range(start, stop, step), default_start=0, default_step=1
#all number must be integers
#the first number of the array built by range will be start
#the last number of the array built by range will be (stop-step)
    
#The following example iss from: http://en.wikibooks.org/wiki/Non-Programmer's_Tutorial_for_Python_3/For_Loops
demolist = ['life',42,'the universe',6,'and',7,'everthing'];
for item in demolist:
    print("The Current item is:",item);
    
#The output is like this:
#The Current item is: life
#The Current item is: 42
#The Current item is: the universe
#The Current item is: 6
#The Current item is: and
#The Current item is: 7
#The Current item is: everything

#while loop
def sequence(n):
    while n!=1: #while n is NOT equal to 1
        if n>1000:
            print("This number is too large");
            break; #terminate the execution of the function
        elif n%2 == 0:
            print(n);
            n = n//2;
        else:
            print(n);
            n = n*3+1;
        
# This concludes today's study.