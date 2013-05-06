# Python 3.3.0 Practice Notes
# Day 5: December 25, 2012

# A tuple is a sequence of values.
# Use parenthesis for tuples, though it is not necessary.
t1 = ('a','b','c','d','e','f');
#to make a tuple with a single element, parenthesis does not work. Use comma.
t2 = 'a',; #<class 'tuple'>
t3 = ('a'); #<class 'str'>
#use tuple() to create a tuple (empty or not).
t4 = tuple();#empty tuple
t5 = tuple('lupins');#tuples of each element of the string
print(t5);
#>>>('l', 'u', 'p', 'i', 'n', 's')
#use bracket to index tuple
t5[3];
#>>>'i'
# In contrast to list, tuples are immutable
t5[3]='A';
#>>>TypeError: 'tuple' object does not support item assignment
#we can reassign the tuple with a new tuple
t5=t5[:3]+('A',)+t5[4:];
print(t5);
#>>>('l', 'u', 'p', 'A', 'n', 's')

# Tuple Assignments
email_addr='monty@python.org';
uName,domName=email_addr.split('@');#splitting the string at '@'
print(uName);
#>>>monty
print(domName);
#>>>python.org

# Tuple as return values
t=divmod(7,3);
print(t);
#>>>(2,1) #(quotient, remainder)
#we may also do
quot,remd=divmod(7,3);
print(quot);
#>>>2
print(remd);
#>>>1
# An example function that returns tuple
def min_max(t):
    return min(t),max(t);

t=(1,2,3,4,5,6,7,8);
t_min,t_max=min_max(t);
print(t_min);
#>>>1
print(t_max);
#>>>8

#'*' in front of the parameter: gather or scatter
#gather: takes arbitrarily many arguments and do commands with all of them
def printAll(*arg):
    print arg; #print every single input arguments

#scatter: given one argument (e.g. tuple), separate them to fit what the command
#requires
t=(7,3);
divmod(t);
#>>>TypeError: divmod expected 2 arguments, got 1
divmod(*t);
#>>>(2,1)

def sumAll(*args): #this should gather all the args into a tuple
    return sum(args); #sums a tuple

sumAll(2,3,4,5,6,2,3,4,1);
#>>>30

# List and tuples
#zip() combines multiple sequences into a list of tuples
s='abc';
t=[0,1,2,3,4];
z=zip(s,t);#note the returned list has length of the shorter sequence
print(z);
#supposedly, it looks like the following, but Python 3 does not print like this
#[('a',0),('b',1),('c',2)] -->Python2
#<zip object at 0x0000000002FBC5C8> -->Python3
for letter,number in z:
    print(letter,number);

#>>>
#a 0
#b 1
#c 2

#to transverse the elements and indices a sequence, use enumerate()
for index, element in enumerate('abc'):
    print(index,element);
#>>>
#0 a
#1 b
#2 c

# Dictionaries and tuples
#.items() method of dictionaries returns a list of tuples, where each element
#of the tuple is a (key,value) pair
d={'a':1,'b':2,'c':3,'d':4};
t=d.items();
print(t);
#>>>dict_items([('d', 4), ('b', 2), ('c', 3), ('a', 1)])
#in fact, this 'dict_items' is called a iterator, but it behaves like a list,
#and we may convert this into a list by doing list(d.items())

#create a dictionary of (string,index)
d=dict(zip('asdfgh',range(len('asdfgh'))));
print(d);
#>>>{'h': 5, 'f': 3, 'g': 4, 'd': 2, 's': 1, 'a': 0}

#.update() method of dictionary adds a list of tuples to the dictionary
d.update([('z',7),('m',9)]);

#use tuples as keys of a dictionary
d.clear();#clear all the items in the dictionary
lastName=['Smith','Wang','Lee','Allen','Georgeton','Schuman'];
firstName=['John','Julie','Thomas','Nich','Busk','Henry'];
phoneNum=['626','232','888','333','123','999'];

d=dict();
for i in range(0,len(lastName)):
    d[lastName[i],firstName[i]]=phoneNum[i];
    
# Tuple comparison
#tuple compares the first elements of each tuple, if tie, go to the next one
#sorting words from shortes to the longest
def sort_by_length(words_list):
    l=list();#empty list for the sorted words
    for word in words_list:
        l.append((len(word),word));
    
    l.sort(reverse=True);#'reverse=True' make sure sorting in descending order
    sorted_list=[];
    for wl,wd in l:
        sorted_list.append(wd);
    
    return sorted_list;

word_list=['adds','vista','banana','fda','joke'];
after_sort=sort_by_length(word_list);
print(after_sort);
#>>>['banana', 'vista', 'joke', 'adds', 'fda']
#note that 'joke' and 'adds' have the same length. It will sort by the second
#element of the tuple, which are the words. Since 'j' comes after 'a', and
#we specified to sort by descending order, 'joke' comes before 'adds'

# When to use tuple
#1) when trying to return a list of parameters in a function
#2) when required using an immutable sequence, for instance, creating the key
#of a dictionary (can also use strings)
#3) when passing a sequence to a function to avoid aliasing

#This concludes today's study.