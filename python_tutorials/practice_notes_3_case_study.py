# Python 3.3.0 Practice Notes
# Day 3: November 25, 2012
# Case Study Script

open_file = open('words.txt');#create a file handle for the text file
open_file.readline();#should return the first line in the document
#>>>'aa\n'
#where \n is the new line delimiter in Windows
#in Linux, it should be \r\n
open_file.readline();#a second time calling .readline() method should read the next line
readLine=open_file.readline();#store the read line in a variable
#readWord=readLine.strip();#this should strip the annoying \n delimiter in Python 2
print(readWord);
#However, in Python 3, it looks like once .readline is stored in a variable
#it automatically strip the delimiter, thus .strip is not available / useless in Python 3

#This concludes the case study