# Python 3.3.0 Practice Notes
# Day 4: December 23, 2012

# Difference between a list and a dictionary
# In Python, the indices of a list must be integers;
# However, in a dictionary, the indices can be almost any type
# lists use square bracket '[]',
# whereas dictionaries use curly braces '{}'

# Build a dictionary that maps English words to Spanish words
eng2sp = dict();#create a new empty dictionary
print(eng2sp);
#>>>{} #empty braces/dictionary
#add elements to the dictionary
eng2sp['one']='uno'; #format: var[key]=value
eng2sp['two']='dos';
eng2sp['three']='tres';
print(eng2sp);
#>>>{'two': 'dos', 'three': 'tres', 'one': 'uno'}
#note the order is messed up and unpredictable somehow
#however, this does not create a problem, because
#the elements in the dictionary is not indexed by order or
#conventional intergers, but in this case, by the string
print(eng2sp['four']);
#>>>KeyError: 'four' #we cannot index something whose key does not exist
len(eng2sp);#find the length of the dictionary
#>>>3
'one' in eng2sp #check if a key 'one' exists in the variable
#>>>True
'uno' in eng2sp #check if a key 'uno' exists in the variable
#>>>False
#to find if a value exists, we have to get the values as a list first
vals=eng2sp.values();#get only the values of the dictionary
'uno' in vals;#check if 'uno' is in the value
#>>>True
#similarly, to convert a dictionary to only keys, use "keys()" method
word_keys=eng2sp.keys();
print(word_keys);
#>>>dict_keys(['two', 'three', 'one']) #<class 'dict_keys', not 'list'
#in general, when we have a large number of items, dictionary is faster than
#lists, as lists use search algorithm, whereas dictionaries use a hashtable
#algorithm (not sure what this means)

#Use dictionary as a set of counters
def histogram(s):
    d = dict(); #create an empty dictionary
    for c in s: #for each element of c in string s
        if c not in d:#if c is not in the dictionary d
            d[c]=1;#take a note of c the key, and count as 1, the first time
        else:
            d[c]+=1;#take a note of c the key, and increase the count
    return d;

#now use this function to count the number of each letter in a word
word_example='brontosaurus';
h=histogram(word_example);
print(h);
#>>>{'n': 1, 'o': 2, 'b': 1, 't': 1, 'u': 2, 'a': 1, 'r': 2, 's': 2}
#again, the returned key is very much random
h.get('f');#get the value with key 'f', otherwise, return 'None' (default)
#>>>0
h.get('o',-1);#get the value with key 'o', otherwise, return '-1'
#>>>2
#We may use get() method to redefine histogram more concisely
def histogram2(s):
    d = dict();
    for c in s:
        d[c]=d.get(c,0)+1;
        #reasoning:
        #if c does not exist in the dictionary, return 0+1=1;
        #if c already exists in the dictionary, return its current_value+1
    return d;

h2=histogram2(word_example);
print(h2);
#>>>{'n': 1, 'o': 2, 'b': 1, 't': 1, 'u': 2, 'a': 1, 'r': 2, 's': 2}

#sort by keys and print dictionary by alphabetical order
def print_dict(d): #void function
    keys_only=list(d.keys());#get the keys as a list
    keys_only.sort();#sort the keys alphabetically
    for e in keys_only:
        print(e,d[e]);
        
print_dict(h2);
#>>>print_dict(h2)
#a 1
#b 1
#n 1
#o 2
#r 2
#s 2
#t 1
#u 2

# Reverse lookup
def reverse_lookup(d,v): #reverse look up a value v in dictionary d
    for e in d:
        if d[e]==v:
            return e;
    raise ValueError('value does not appear in the dictionary');
                    #if eventually there is nothing to return
                     #give out an error message 'ValueError:...'

#successful reverse lookup
reverse_lookup(h2,2);
#>>>'o'
#failed reverse lookup
reverse_lookup(h2,5);
#Traceback (most recent call last):
#  File "<console>", line 0, in <module>
#  File "<console>", line 0, in reverse_lookup
#ValueError: value does not appear in the dictionary


# List can be values in a dictionary, but cannot be keys
# Inverting Dictionary
def invert_dict(d):
    d_inverse=dict();
    for k in d: #get the key from d
        val = d[k];#temporarily store the value at key
        if val not in d_inverse:
            d_inverse[val]=[k];#store the key (new value) as a list, since there
                             #keys with the same value
        else:
             d_inverse[val].append(k);
    return d_inverse;

#example
h3=histogram2('parrots');
print(h3);
#>>>
h3_inverse=invert_dict(h3);
print(h3_inverse);

# Definition: A 'hash' is a function that takes a value (of any type) and
# returns as an integer. Dictionaries use these integers, called hash values, to
# store and look up key-value pairs. Therefore, keys must be immutable, and
# since list is mutable, it cannot be keys.

#The following is a more concise version of invert_dict
def invert_dict2(d):
    d_inverse=dict();
    for k in d: #get the key from d
        val = d[k];#temporarily store the value at key
        d_inverse.setdefault(val,[]).append(k);
        #reasoning: very similar to histogram2
        # if key 'val' does not exist in the inverted dictionary
        # return [].append(k) to start a new key at 'val', and assign value 'k'
        # if val already eixsts
        # append another  value 'k' at key 'val'
    return d_inverse;

# Memo
known={0:0,1:1};#dictionary of first two fibonacci numbers
def fibonacci(n):
    if n in known:#if the number is already known
        return known[n];#reutrn from the memo
    res=fibonacci(n-1)+fibonacci(n-2);#if not, recalculate
    known[n]=res;#store the new calculation in the memo
    return res;#return calculated

# Global vs. Local variables
been_called = False; #This is a global variable, belong to __main__ frame
def example_1():
    been_called = True; #This creates a new local variable of the same name

def example_2():
    global been_called;#This will call the global variable declared previously
    been_called=True; #now, we can reassign the global variable
    
#These variables mentioned above are immutable variables, however, if the
#variable is mutable, then we can reassign the values without redeclaring
known = {0:0,1:1};
def example_3():
    known[2]=1;

#However, to reassign the entire variable, we have to redeclare it
def example_4():
    global known;
    known = dict();
    
# Long integer in Python 2
# if an integer is very long, it is stored as type 'long' instead of type 'int'.
# this only happens in Python 2, as Python 3 stores long integers as type 'int'.

#This concludes today's study.