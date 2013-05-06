# Python 3.3.0 Practice Notes
# Day 8: January 19, 2013

# Card object
class Card(object):
    """Represents a standard playing card."""
   
    def __init__(self,suit=0,rank=2):#define default card: Club of 2
        """Define a Card object
         #suit number:  Spades    -->3
                        Hearts    -->2
                        Diamonds  -->1
                        Clubs     -->0
        """
        self.suit=suit;
        self.rank=rank;
    
    suit_names=['Clubs','Diamonds','Hearts','Spades'];
    rank_names=[None,'Ace','2','3','4','5','6','7','8','9',
                '10','Jack','Queen','King'];#Shift+Enter to break the line
    def __str__(self):#define print
        """Allows printing"""
        return '%s of %s' %(Card.rank_names[self.rank],
                            Card.suit_names[self.suit]);
    
    #enable orderablility of the object
    def __lt__(self,other): #less than
        return (self.suit,self.rank)<(other.suit,other.rank);        
    def __le__(self,other): #less than or equal to
        return (self.suit,self.rank)<=(other.suit,other.rank);        
    def __eq__(self,other): #equal
        return (self.suit,self.rank)==(other.suit,other.rank);         
    def __ge__(self,other): #greater than or equal to
        return (self.suit,self.rank)>=(other.suit,other.rank);       
    def __gt__(self,other): #greater than
        return (self.suit,self.rank)>=(other.suit,other.rank);
    def __ne__(self,other): #not equal to
        return (self.suit,self.rank)!=(other.suit,other.rank);
    
    def keyFunction(self):
        return (self.suit,self.rank);

# Test class Cards
#create a card
queen_of_diamonds=Card(1,12);
#print(queen_of_diamonds);
king_of_clubs=Card(0,13);
king_of_clubs.compareCards(queen_of_diamonds)
#>>-1
C=Card(0,13);
king_of_clubs.compareCards(C);
##>>0

# Deck Object
import random;

class Deck(object):
    def __init__(self):
        """Create a 52 card deck card"""
        self.cards=[];#empty list with object Cards
        for suit in range(4):#enumerates 0 to 3
            for rank in range(1,14):#enumerates 1 to 13
                card = Card(suit,rank);
                self.cards.append(card);
    
    def __str__(self):
        res=[];
        for card in self.cards:
            res.append(str(card));#str(card) converts the card to string
                                  #instead of displaying its number
        return '\n'.join(res);
    
    def pop_card(self):
        return self.cards.pop();#remove the  last card from the list
                                #and return it
    
    def add_card(self,card):
        self.cards.append(card);#append a card to the list/deck
        
    def shuffle(self):
        random.shuffle(self.cards);#randomly shuffle the list/deck
        
    def sort(self, order=1):
        """sort order-->1:ascending, 0:desending"""
        cardKeys=[];
        for card in self.cards:
            cardKeys.append((card.suit,card.rank));#get keys of each card
        
        if order ==1:
            cardKeys.sort(reverse=False);
        else:
             cardKeys.sort(reverse=True);
             
        del self.cards;#clear cards
        self.cards=[];#restart a new deck
        for key in cardKeys:
            card=Card(key[0],key[1]);#creating new cards
            self.cards.append(card);#creating the new deck in order
        

#Test Deck object
deck = Deck();
print(deck);
#Ace of Clubs
#2 of Clubs
#...
#10 of Spades
#Jack of Spades
#Queen of Spades
#King of Spades
deck.shuffle();
print(deck);
deck.sort(0);#sort the deck descending order
print(deck);

# Hand object and inheritance
class Hand(Deck):#Hand inherits Deck
    """Represents a hand of playing cards."""
    def __init__(self,label=''):
        self.cards=[];
        self.label=label;
    
    def move_cards(self,hand,num):
        """Draw num cards from the one Hand/Deck
        and give it to another Hand/Deck"""
        for i in range(num):
            hand.add_card(self.pop_card());
    
#Test Hand object
hand=Hand('New Hand');
#hand inherits whatever is in deck object
deck=Deck();#get a new deck
card=deck.pop_card();#take a card out from the Deck
hand.add_card(card);#adding a card from a Deck to the Hand
print(hand);
#>>>King of Spades

#This concludes today's study.










