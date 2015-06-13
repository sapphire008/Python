# Python 3.3.0 Practice Notes
# Day 7: January 4, 2013

##############################################################################
# Class and Object: user defined types
class Point(object): #notice that "class" and "object" are keywords
    """Represents a point in 2D space.""" #annotation
    pass;
    
print(Point);
#>>><class '__main__.Point'>

#an instance of a class
blank=Point();
print(blank);
#>>><__main__.Point object at 0x0000000003130860>

#Assigning attritubes to the class:
#This is very similar to structures in MATLAB
#the following assign x and y attribute (in MATLAB, fields) to instance blank
blank.x=3.0;
blank.y=4.0;
print("X-coordinate:",blank.x);
print("Y-coordinate:",blank.y);
#>>>X-coordinate: 3.0
#>>>Y-coordinate: 4.0
#we may also do this:
print('(%g,%g)' %(blank.x,blank.y));
#>>>(3,4)
#it is also possible  to call functions and methods with attributes
import math;
distance=math.sqrt(blank.x**2+blank.y**2); #note ** replaces ^ in Python 3
print(distance);
#>>>5.0;

def distance_between_points(p1,p2):
    """take in two Points objects and calculate their distance"""
    import math;
    distance=math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2);
    return distance;

point_1=Point();
point_1.x=2.3;
point_1.y=3.6;
point_2=Point();
point_2.x=10.2;
point_2.y=15.3;

DIST=distance_between_points(point_1,point_2);
print(DIST);
#>>>14.117365193264641

def print_point(p):
    print("(%g,%g)" %(p.x,p.y));


# Rectangle, with Points embedded
class Rectangle(object):
    """Represents a rectangle.
    attributes: width, height, corner.
    """
    pass;

box=Rectangle();
box.width=100.0;
box.height=200.0;
box.corner=Point();#Point object is embedded within Rectangle instance
box.corner.x=0.0;
box.corner.y=0.0;

#instance can be a return value
def find_center(rect):
    p=Point();
    p.x=rect.corner.x+rect.width/2.0;
    p.y=rect.corner.y+rect.height/2.0;
    return p;

center=find_center(box);
print_point(center);
#>>>(50,100)

#Customized objects are mutable
print(box.width);
#>>>100.0
box.width=box.wdith+10;
print(box.width);
#>>>110.0

#Since they are mutable, there may be potentially problems wth aliasing
#however, there si a module "copy" we can use to duplicate the object
import copy;

box2=copy.copy(box); #shallow copy, which does not copy the embedded elements

box is box2;
#>>>False
box==box2;
#>>>False #because in object, "==" operator is the same as "is" operator

#Also, the shallow copy does not copy the embeded objects
box.corner is box2.corner;
#>>>True

#to do a deep copy, use copy.deepcopy
box3=copy.deepcopy(box);
box.corner is box3.corner;
#>>>False

#if uncertain what attributes that an object have, use hasattr(object,'attr');
hasattr(box,'x');
#>>>False
hasattr(box,'corner');
#>>>True

##############################################################################
# Class and Function
class Time(object):
    """Represents the time of the day.
    attributes: hour, minute, second
    """
    pass;

time=Time();
time.hour=11;
time.minute=59;
time.second=30;

def print_time(t):
    print('%.2d:%.2d:%.2d' %(t.hour,t.minute,t.second));
#note that %.2d prints 2 digits

#Pure functions and Modifiers:
def add_time(t1,t2): #pure function
    """Adding two time"""
    SUM = Time();
    SUM.hour=t1.hour+t2.hour;
    SUM.minute=t1.minute+t2.minute;
    SUM.second=t1.second+t2.second;
    return SUM;

#pure function does not modify any of the objects passed onto its arguments
#in this case, t1 and t2 are not changed at all

#Test the function
start=Time();#specifying start time
start.hour=9;
start.minute=45;
start.second=0;

duration=Time();#specifying duration
duration.hour=1;
duration.minute=35;
duration.second=0;

endTime=add_time(start,duration);#calculating end time
print_time(endTime);#print end time
#>>>10:80:00

#however, this is not what we expected for time in real life, therefore, we
#need modifier functions
def increment(time,seconds):#a modifer function changes its input
    time.second+=seconds; #increase the time by specified seconds
    
    if time.second>60:#if second greater than 60
        time.minute+=time.second//60;#increase minute by quotient
        time.second=time.second%60;#find the remainder after dividing 60
        
    if time.minute>=60:
        time.hour+=time.minute//60;
        time.minute=time.minute%60;
#we may also invoke a recursion in the function, but it may be less efficient

increment(endTime,0);
print_time(endTime);
#>>>11:20:00

# Prototype vs. Patch: write, test, and retest to correct errors
#we can either write a pure function that includes all the algorithms,
#or we can create different parts of that function by creating simpler
#individual functions which can be called into another function that
#carries out the goal. This is called planned development, which usually
#involves high-level insights that breaks down the problem.

##############################################################################
# Class and Method:
#Difference between method and function:
#1). Methods are defined inside a class in order to make the relationship
#between class and the method clear
#2). They syntax for invoking a method is different from the syntax for calling
#a function

#To create a method inside a class is like create a function, except it is
#under the class object, rather than the __main__
class Time(object):
    def print_time(time):#this first parameter of the method is usually called
                        #self, so we may use "self" instead of "time"
        print('%.2d:%.2d:%.2d' %(time.hour,time.minute,time.second));

#Testing
StartTime=Time();
StartTime.hour=2;
StartTime.minute=34;
StartTime.second=31;
Time.print_time(StartTime);#now print_time is a method of Time
#>>>02:34:31
#we can also use method syntax to get the same result
StartTime.print_time();
#>>>02:34:31
#in this case, "StartTime" is the subject with method "print_time"

#We now creates several methods for class Time. Note that it is important
#to leave NO empty line between each method, at least in Komodo.
def int_to_time(seconds):
        """convert seconds in integer to a time object"""
        time=Time();
        minutes,time.second=divmod(seconds,60);
        time.hour,time.minute=divmod(minutes,60);
        return time;
#the reason not to put this function inside Time as a method: the input is
#an integer, not a Time object.

class Time(object):
    def print_time(self):
        """print time object"""
        print('%.2d:%.2d:%.2d' %(self.hour,self.minute,self.second));
    def time_to_int(self):
        """convert a time object to integer"""
        minutes=self.hour*60+self.minute;
        seconds=minutes*60+self.second;
        return seconds;
    def increment(self,seconds):
        """increase a time object by a specified seconds"""
        seconds+=self.time_to_int();
        return int_to_time(seconds);
    def is_after(self,other):
        """check if a time is after another time"""
        return self.time_to_int()>other.time_to_int();
    def __init__(self,hour=0,minute=0,second=0):
        """__init__ method initilize the object with default values"""
        self.hour=hour;
        self.minute=minute;
        self.second=second;
    def __str__(self):
        """convert the object to a string. This allows the object to be
        printed directly using 'print'. """
        return '%.2d:%.2d:%.2d' %(self.hour,self.minute,self.second);
    def add_time(self,other):
        """allows the addition of two times given"""
        seconds=self.time_to_int()+other.time_to_int();
        return int_to_time(seconds);
    def __add__(self,other):#this __add__ method checks type of "other"
        """adds time together"""
        if isinstance(other,Time):
            return self.add_time(other);
        elif isinstance(other,int):
            return self.increment(other);
    def __radd__(self,other):
        """gives communitative property of addition to the class object"""
        return self.__add__(other);

#testing
start=Time();
start.hour=1;
start.minute=32;
start.second=41;
end=Time();
end.hour=2;
end.minute=34;
end.second=24;
end.is_after(start);#chekc to see if end time is after start time
#>>>True

#testing __init__ method
time=Time();
time.print_time();
#>>>00:00:00
time=Time(9);
time.print_time();
#>>>09:00:00
time=Time(9,30);
time.print_time();
#>>>09:30:00
time=Time(9,30,42);
time.print_time();
#>>>09:30:42

#testing __str__ method
time=Time(9,45);
print(time);#"print" invokes "__str__" method
#>>>09:45:00

#testing __add__ method
start=Time(9,45);
duration=Time(1,35);
print(start+duration); #the "+" should invoke "__add__" method
#>>>11:20:00
duration=30;#30 seconds of duration
print(start+duration);
#>>>09:45:30
#however, the addition is not communitative
print(duration+start);
#>>>TypeError: unsupported operand type(s) for +: 'int' and 'Time'
#this can be solved using __radd__ or "right_side add"
#it is invoked when the Time object is appears on the right side of the
#"+" operator
#after adding __radd__ method, try add again
start=Time(9,45);
duration=30;
print(duration+start);
#>>>09:45:30;

# Polymorphism: functions that can work with several types
#for example, sum() is polymorphic and adds up the objects as long as the object
#itself supports addition
t1=Time(7,43);
t2=Time(7,41);
t3=Time(7,37);
total=sum([t1,t2,t3]);
print(total);
#>>>23:01:00

#Use __dict__ method to print out a dictionary of attributes and values
print(t1.__dict__);
#>>>{'hour': 7, 'minute': 43, 'second': 0}

#This concludes today's study.