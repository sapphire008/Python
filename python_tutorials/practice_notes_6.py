# Python 3.3.0 Practice Notes
# Day 6: December 29, 2012

# Write a file
fout=open('output.txt','w');#'w' specifies the file can be overwritte
print(fout)
#>>><_io.TextIOWrapper name='output.txt' mode='w' encoding='cp936'>
line_1="This here's the wattle,\n";#\n starts a new line
fout.write(line_1);#writes line_1 to the first line of the txt file
#note that for method .write(input), the input must be a string
line_2="the emblem of our land.\n";
fout.write(line_2);#this will append line_2 to the second line of the txt file
fout.close();#this close the file

#format operators and sequence%: %first operand %second operand
#   '%d': things that follow this (second operand) should be a integer.
          #even if the specified second operand is a decimal, it will convert
          #the decimal into a integer by truncating the number after decimal
camels=42; #integer
'%d' %camels #%format sequence(first operand) %second operand
#>>>'42'
#the place where this format sequence can appear anywhere
A='I have spotted %d camels' %camels;
print(A);
#>>>I have spotted 42 camels

#   '%g': things that follow this (second operand) will be formatted to decimal
#   '%s': second operand will be formatted as a string

A='In %d years, I have spotted %g %s.' %(3,0.1,'camels');
print(A);
#>>>In 3 years, I have spotted 0.1 camels.
#note that the nubmer of format sequence has to match the number of elements
#in the tuple. The types also have to match

# File names and paths
import os; #importing module for working with files and dir
current_working_directory=os.getcwd();
print(current_working_directory);
#>>>C:\Users\Edward\Documents\Assignments\Python
#>>><class 'str'>
os.path.abspath('output.txt');#find absolute path
#>>>'C:\\Users\\Edward\\Documents\\Assignments\\Python\\output.txt'
os.path.exists('output.txt');#check if some path exists
#>>>True
os.path.isdir('output.txt');#check if a path is a directory
#>>>False
os.path.isfile('output.txt');#check if a path is a file
#>>>True
os.listdir(current_working_directory);#returns a list of files in cwd
#>>>['output.txt', 'practice_notes_1.py', 'practice_notes_6.py']
#example function
def walk(dirname):
    for name in os.listdir(dirname):
        path=os.path.join(dirname,name);#join dir and file name
        if os.path.isfile(path) #if the one we have is a file
            print(path); #print the file name
        else: #if it is a directory
            walk(path);#otherwise, walk in the subdirectory given by file

# Catching exceptions
try: #try to do the following
    fin = open('bad_file');
    for line in fin:
        print(line);
    fin.close();
except: #like catch in MATLAB, do the following if error occurs
    print('Something went wrong');
#Exercise 14.2
def sed(strPat,strRep,fin,fout):
    try:
        fin_open=open(fin,'r');
        fout_open=open(fout,'w');
        for line in fin_open:
            line.replace(strPat,strRep)
            fout_open.write(line);
        fin_open.close();
        fout_open.close();
    except:
        print(fin,'does not exist!');
            

# Database
import anydbm;#for managing database
db=anydbm.open('captions.db','c');#'c' for creating the database if not exists
#The database should work like a dictionary
db['cleese.png']='Photo of John Cleese.';
#many dictionary methods, such as .key(), ,value() also work on database
#keys and values must both be strings
#after modifying the database, we must close it
db.close();

# Pickling
#pickle is a module that can convert any object into a string, then store in
#the database; it is also able to convert the string back to object
import pickle;
t=[1,2,3];
s=pickle.dumps(t);#dump the object 't' into a string 's'
print(s);
#>>>b'\x80\x03]q\x00(K\x01K\x02K\x03e.'
#though not making much sense to human
t_back=pickle.loads(s);#convert the string back to object
print(t_back);
#>>>[1, 2, 3]
#note that t and t_back have the same value, but they are not the same object
#it has the same effect as copying the object
t==t_back;
#>>>True
t is t_back;
#>>>False
#shelve module will incoporate both anydbm and pickle that it converts any
#object to strings to store in the database, and retrieve them by converting
#the stored strings back to the object. It appears as if the object is stored
#as is in the database

# Pipe
import subprocess;#a newer module that replaces os

# Modules
#Modules are simply .py scripts with a bunch of functions
#to prevent the modules (scripts) execute itself, we may enclose the protion
#of the code that gives output with
if __name__=='__main__';#note there are two underlines before and after both
                        #'name' and 'main'. Also, '__name___' is built-in
                        #variable
    print('whatever we want');

#this means, when we try to run the script from a shell, the variable
#__name__ has a value of '__main__', whereas when we import the script as a
#module, the script should not have that value
#usually, what is being enclosed are the test scripts of each function

#Important note: if the module has already been imported, calling import will
#do nothing, even if the module scripts have been changed after the first import
#the best way is to restart the program and reimport everything

#This concludes today's study.