# Python 3.3.0 Practice Notes
# Day 3: November 25, 2012

# Strings
#Note for defining a string
#I personally perfer using double quotes, since single quotes are not good
#when the string contains possessive- or contraction-like words, for instance
#>>>ABCD='I'd like this to go';
#>>>  File "<console>", line 1
#>>>    ABCD='I'd like this to go';
#>>>            ^
#>>>SyntaxError: invalid syntax
#However, it is legal to do
#>>>ABCD="I'd like this to go";
#Nonetheless, if the string do contain double quotation mark, it looks like
#we still have to switch to single quotation mark.
#The question would be what if the string contains both double and single quotation mark?
#That is, how to define a string with such sentence: I'd like to ask "him" about it.
#One way, perhaps is to do concatenation
a_word="asdfghjkl;";
for i in range(0,len(a_word),1):
    print(i,":",a_word[i]);#this should print out each letter of the string stored in a_word, forward

#This can be done in another way
for i in a_word:
    print(i);#this should print out each letter of the stringg stored in a_word, forward

for j in range(1,len(a_word)+1,1):
    print(-j,":",a_word[-j]);#this should print out each letter of the string stored in a_word, backwards

#String Indexing
a_word[0:3];#or equivalently
a_word[:3];
#both lines should print out the string up to its index 3-1 (total 3 letters)
#>>>'asd'
a_word[3:len(a_word)];#or equivalently
a_word[3:];
#both lines should print out the string from its index 3 to the end (total len(a_word)-3 letters)
#>>>'fghjkl;'
a_word[:];#this should print out the whole string, equivalent to print a_word directly
#Important Note: unlike MATLAB, string in Python are not treated as a matrix/vector
#Strings in Python is immutable, meaning its elements cannot be changed
#Therefore, it will be an error to write
#a_word[3]="K";
#>>>TypeError: 'str' object does not support item assignment

#String Method
b_word = "banana";
new_b_word = b_word.upper();#this "S.upper" method converts all lowercase letter to uppercase letter
print(new_b_word);
#>>>'BANANA'
A_index = b_word.find('an',3,10);
print(A_index);
#this "S.find(substring,start,end)" method should find the lowest index of specified substring
#notice that even if the end index exceeds the length of the string,
#unlike MATLAB, there will be no message indicating that index exceeds dimension
#if there is no such substring within the searched string, returns -1
the_word="BANANANa";
the_word.isupper();#returns true if the string ONLY contains uppercase letter
#>>>False

#Another note on the notation of help documentation/calltips of Python functions and methods
#for instance, S.find(sub[, start[, end]])
#The notation indicates that "sub" (since it is outside the bracket) is required,
#whereas "start" is optional (since it is inside a bracket).
#However, once start is specified, "end" is now optional
#In another words, "end" cannot be specified without "start"

#The "in" Operator
#This operator checkes if the string specified before the "in" operator
#is the substring of the string specified after the "in" operator
"a" in "banana";
#>>>True
"seed" in "banana";
#>>>False

#String Comparison
"A"<"B" and "B"<"C"
#>>>True
#Both statements are true, which makes the entire line true
#Strings are compared directly as numbers
#the number that each character corresponds to ASCII
"a">"A"
#>>>True
"b"<"B"
#>>>False
# Just like Java, when comparing strings with multiple letters,
# Python compares the first letter of each word, if they are the same,
# Python goes to the second letter, and then compare them.
# A list of words can be organized in such a way
######################################################################################################################

# Lists
List_A=["asdf","jkl;","Such a good weather"];#list of strings
List_B=[1,3,4,12,234];#list of integers
List_C=[];#empty list
List_Mixed=[1,2.3424,"sanskrit",23,"floating above", 3.242,"12.23"];#mixed different types
#Lists are mutable
List_Mixed[2]='not a sanskrit';
#>>>[1, 2.3424, 'not a sanskrit', 23, 'floating above', 3.242, '12.23']

#"in" operator for lists
"sanskrit" in List_Mixed
#>>>False
"not a sanskrit" in List_Mixed
#>>>True

#Nested Lists
List_nested=["good",1.234,["bad",3.1234,32],[2,3,4,5,6]];
List_nested[2];#return the second element of the list List_nested
#>>>['bad', 3.1234, 32]
List_nested[2][1];#call the index 2 element of the list List_nested, then from the returned element, call its index 1 element
#>>>3.1234

#List Operatoions
a=[1,2,3];
b=[4,5,6];
c=a+b;#concatenating a and b
print(c);
#>>>[1,2,3,4,5,6]
d=a*4;#repeat a 4 times in the new list
print(d);
#>>>[1,2,3,1,2,3,1,2,3]

#List indexing--very similar to string indexing
f=c[:3];
print(f);
#>>>[1, 2, 3]
t=c[3:];
print(t);
#>>>[4, 5, 6]
s=c[:];
print(s);
#>>>[1, 2, 3, 4, 5, 6]

#List Methods
t=['a','b','c'];
t.append('d');#appending another element to the end of the list, void method
print(t);
#>>>['a', 'b', 'c', 'd']
#compare to
t.append(['e','f','g']);
print(t);
#>>>['a', 'b', 'c', 'd', ['e', 'f', 'g']]
#To append each element of another list to a list, use extend
t1=['a','b','c'];
t2=['d','e','f','g'];
t1.extend(t2);#appending each element of t2 to t1, void method
print(t1);
#>>>['a', 'b', 'c', 'd', 'e', 'f', 'g']
t=['adf','gdasdf','deas','adsff','ggas'];
t.sort();#void method
print(t);#sort the list
#>>>['adf', 'adsff', 'deas', 'gdasdf', 'ggas']

#Map, filter, and reduce
#one way to sum up all the elements in the list
def add_all(t):
    total=0;
    for x in t:
        total+=x;#same as JAVA, equivalent to total  = total+x;
    return total

t=[1,2,3,4,5];
sum_all=add_all(t);
print(sum_all);

#A simpler way to add all elements is using sum()
sum(t);
#Reduce: an operation that combines all the element in a list into a single value
#accumulator: a variable that accumulates the result of each iteration when transversing through a list
#map: an operation that "maps" a function to each element in a sequence

#Deleting elements
#If we know the index of the element
t=['a','b','c','d','e'];
x=t.pop(1);#returns the element being deleted, and modify t after deleting
#List.pop([index]), default_index is the index of the last element
print(t);
#>>>['a', 'c', 'd', 'e']
print(x);
#>>>b
#using del() operator gives the same effect
t=['a','b','c','d','e'];
del(t[1:3]);#delete up to but not including index 3 elements, so, only index 1 and 2 are deleted
print(t);
#On the other hand, if we know the element itself but not the index of it
t=['a','b','c','d','e'];
t.remove('b');#void method
print(t);

#converting between lists and strings
s="spam";
t=list(s);#convert each letter of s into a list of letterz
print(t);
#>>>['s', 'p', 'a', 'm']
s="pining for the fjords";
t=s.split();#S.split([sep [,maxsplit]]), default_sep = " " space, can set maximum number of split
print(t);
#>>>['pining', 'for', 'the', 'fjords']
t=['pining', 'for', 'the', 'fjords'];
delimiter=" ";
s=delimiter.join(t);#join the list of words with delimiter
print(s);

# Objects and values
a="banana";
b="banana";
a is b;#checks if two objects are identical
#>>>True
#This means a and b are the same objects, and of course, with the same value
#However,
a=[1,2,3];
b=[1,2,3];
a is b;#checks if two objects are identical
#>>>False
#This means that even though a and b have the same value, they are different objects
#Instead, list a and list b are called "equivalent", whereas string a and string b are called "identical"

#In comparison
a=[1,2,3];
b=a;
a is b;
#>>>True
#We call a is being aliased by b
#if the aliased object is mutable, then change of the aliased object affects all of its alias
a[2]=100;
print(a);
#>>>[1, 2, 100]
print(b);
#>>>[1, 2, 100]
#Notice that b is also changed even though we did not modify it
#This is very different from MATLAB!!!
#Thi must be noted carefully when coding, since it is so error prone
#The question is how to work around this. Look at the following example for some hints

def give_tail(t):
    return t[1:];#this returns a NEW list, with the original t unmodified

t=[1,2,3,4,5,6];
s=give_tail(t);
print(t);
#>>>[1, 2, 3, 4, 5, 6]
print(s);
#>>>[2, 3, 4, 5, 6]

#The wrong way to define the function
def bad_give_tail(t):
    t=t[1:];#trying to reassign t, but t does not change.

t=[1,2,3,4,5,6];
s=bad_give_tail(t);
print(t);
#>>>[1, 2, 3, 4, 5, 6]
print(s);
#>>>None
#In another word, without the "return" statement, the function is a void function
#On the contraray, MATLAB does a better job at this, without the need to worry about aliasing.

#To create copies of the original list, use this:
t=[1,2,3,4,5,6];
original_list=t[:];#create a copy of the original list, without aliasing
t is original_list;#test to see if they are the same list
#>>>False
t.append(7);
print(t);
#>>>[1, 2, 3, 4, 5, 6, 7]
print(original_list);
#>>>[1, 2, 3, 4, 5, 6]

#This concludes today's study.