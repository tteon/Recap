

[medium](https://towardsdatascience.com/solid-coding-in-python-1281392a6a94)

The SOLID principles are:

- The Single-Responsibility Principle (SRP)
- The Open-Closed Principle (OCP)
- The Liskov Substitution Principle (LSP)
- The Interface Segregation Principle (ISP)
- The Dependency inversion Principle (DIP)

These five principles are not a specific ordered list (do this, then that, etc) but a collection of best practices, developed through the decades. They are gathered into an acronym, as a mnemonic  vehicle to be remembered, similar to others in computer science, e.g.: **DRY**: *Don’t Repeat Yourself*; **KISS**: *Keep It Small and Simple*; as pieces of accumulated wisdom. A little side note, the acronym was  created years after these five principles were set together.

In general, the SOLID principles are basic learning steps for every code  developer but are usually ignored by those who does not consider the  highest quality of code their absolute priority.

However, as a Data Scientist myself, I have seen that following these principles is beneficial. Specifically, it improves testability, reduces technical debts, and the time needed to implement changes when new requirements  from clients/stockholders arise.

In the following article, I want to explore these five principles and  offer some examples in Python. Usually, the SOLID principles are applied in the context of object-oriented design (i.e.: Python’s classes), but I believe they are valid regardless of the level, and I would like to  keep the example and explanation here, to a level for an “advanced  beginner”, overseeing formal definition.

# 1) The Single-responsibility principle (SRP)

> *“A class should have one, and only one, reason to change”*

In other words, every component of your code (in general a class, but also a function) should have *one and only one responsibility*. As a consequence of that, there should be only a reason to change it.

Too often you see a piece of code that takes care of an entire process all  at once. I.e., A function that loads data, modifies and, plots them, all before returning its result.

Let’s take a simpler example, where we have a list of number L = [n1, n2, …,  nx] and we compute some mathematical functions to this list. For  example, compute the mean, median, etc.

A bad approach would be to have a single function doing all the work:

```python
import numpy as np

def math_operations(list_):
    # Compute average
    print(f'the mean is {np.mean(list_)}')
    # Compute Max
    print(f'the max is {np.max(list__)}')
    
math_operations(list_ = [1,2,3,4,5])
```



The first thing we should do, to make this more SRP compliant, is to split the function math_operations into **atomic functions**! Thus, when a function’s responsibility cannot be divided into more subparts.

The second step is to make a single function (or class), generically named, “main”. This will call all the other functions one-by-one in a  step-to-step process.

```python
def get_mean(list_):
    '''compute mean'''
    print(f'the mean is {np.mean(list_)}')
    
def get_max(list_):
    '''compute max'''
    print(f'the max is {np.max(list_)}')

def main(list_):
    # Compute Average
    get_mean(list_)
    # Compute Max
    get_max(list_)
    
main([1,2,3,4,5])
```



Now, you would only have one single reason to change each function connected with “main”.

The result of this simple action is that now:

1. It is easier to localize errors. Any error in execution will point out to a smaller section of your code, accelerating your debug phase.
2. Any part of the code is reusable in other section of your code.
3. Moreover and, often overlooked, is that it is easier to create testing for each  function of your code. Side note on testing: You should write tests  before you actually write the script. But, this is often ignored in  favour of creating some nice result to be shown to the stakeholders  instead.

This is already a much bigger improvement with respect to the first code  example. But, having created a “main” and calling functions with single  responsibility is not the full fulfilment of the SR principle.

Indeed, our “main” has many reasons to be changed. The class is actually fragile and hard to maintain.

To solve that, let’s introduce the next principle:

# 2) The Open–closed principle (OCP)

> “*Software entities … should be open for extension but closed for modification”*

In other words: You should not need to modify the code you have already  written to accommodate new functionality, but simply add what you now  need.

This does not mean that you cannot change your code when the code premises  needs to be modified, but that if you need to add new functions similar  to the one present, you should not require to change other parts of the  code.

To clarify this point let’s refer to the example we saw earlier. If we  wanted to add new functionality, for example, compute the median, we  should have created a new method function and add its invocation to  “main”. That would have added an *extension* but also *modified* the main.

We can solve this by turning all the functions we wrote into subclasses of a class. In this case, I have created an abstract class called  “Operations” with an abstract method “get_operation”. (Abstract classes  are generally an advanced topic. If you don’t know what an abstract  class is, you can run the following code even without).

Now, all the old functions, now classes are called by the __subclasses__()  method. That will find all classes inheriting from Operations and  operate the function “operations” that is present in all sub-classes.

```python
import numpy as np
from abc import ABC, abstractmethod

class Operations(ABC):
    '''Operations'''
    @abstractmethod
    def operation():
        pass
    
class Mean(Operations):
    '''Compute Mean'''
    def operation(list_):
        print(f'The mean is {np.mean(list_)}')

class Max(Operations):
    '''Compute Max'''
    def operation(list_):
        print(f'The max is {np.max(list_)}')
    
class Main:
    '''Main'''
    @abstractmethod
    def get_operations(list_):
        # __subclasses__ will found all classes inheriting from Operations
        for operation in Operations.__subclasses__():
            operation.operation(list_)
            
if __name__ == "__main__":
    Main.get_oeprations([1,2,3,4,5])
```



If now we want to add a new operation e.g.: median, we will only need to  add a class “Median” inheriting from the class “Operations”. The newly  formed sub-class will be immediately picked up by __subclasses__() and  no modification in any other part of the code needs to happen.

The result is a very flexible class, that requires minimum time to be maintained.

# 3) The Liskov substitution principle (LSP)

> *“Functions that use pointers or references to base classes must be able to use objects of derived classes without knowing it”*

Alternatively, this can be expressed as “*Derived classes must be substitutable for their base classes*”.

In (maybe) simpler words, if a subclass redefines a function also present  in the parent class, a client-user should not be noticing any difference in **behaviour,** and it is a **substitute** for the base class. 
For example, if you are using a function and your colleague change the base class, you should not notice any difference in the function that you  are using.

Among all the SOLID principle, this is the most abstruse to understand and to explain. For this principle, there is no standard “template-like”  solution where it must be applied, and it is hard to offer a “standard  example” to showcase.

In the most simplistic way, I can put it, this principle can be summarised by saying: 
 If in a *subclass*, you redefine a *function* that is also present in the *base class*, the two functions ought to have the same behaviour. This, though, does not mean that they must be *mandatorily* equal, but that the user, should expect that the same *type* of result, given the same input. 
In the example ocp.py, the “operation” method is present in the subclasses and in the base class, and an end-user should expect the same behaviour from the two.

The result of this principle is that we’d write our code in a consistent  manner and, the end-user will need to learn how our code works, only  one.

## Addendum:

(You can skip to the next principle).

A consequence of LSP is that: the new redefined function in the sub-class should be valid and be possibly used wherever the same function in the  parent class is used.

This is not, typically the case, indeed usually we, human, think in terms of set theory. Having a class that define a concept and sub-classes that  expand the first with an exception or different behaviour.

For example, the sub-class “Platypus”, of the base class “Mammals”, would  have the exception that these mammals lay eggs. The LSP, tell us that it would create a function called “give_birth”, this function will have  different behaviour for the sub-class Platypus and the sub-class Dog.  Therefore, we should have had a more abstract base class than Mammals  that accommodate this. 
If this sounds very confusing, do not worry,  the application of this latter aspect of the LSP is rarely fully  implemented, and it rarely leaves the theoretical textbooks.

# 4) The Interface Segregation Principle (ISP)

> “*Many client-specific interfaces are better than one general-purpose interface*”

In the contest of classes, an interface is considered, *all the methods and properties* “**exposed**”, thus, everything that a user can interact with that belongs to that class.

In this sense, the IS principles tell us that a class should only have the interface needed (SRP) and avoid methods that won’t work or that have  no reason to be part of that class.

This problem arises, primarily, when, a subclass inherits methods from a base class that it does not need.

Let’s see an example:

```python
import numpy as np
from abc import ABC, abstractmethod

class Mammals(ABC):
    @abstractmethod
    def swim() -> bool:
        print('Can Swim')
   	
    @abstractmethod
    def walk() -> bool:
        print('Can Walk')
        
class Human(Mammals):
    def swim():
        return print('Humans can swim')
    
    def walk():
        return print('Humans can walk')

class Whale(Mammals):
    def swim():
        return print('Whales can swim')
```



For this example, we have got the abstract class “Mammals” that has two  abstract methods: “walk” and “swim”. These two elements will belong to  the sub-class “Human”, whereas only “swim” will belong to the subclass  “Whale”.

And indeed, if we run this code we could have:

```python
Human.swim()
Human.walk()

Whale.swim()
Whale.walk()

# Humans can swim
# Humans can walk
# Whales can swim
# Can Walk
```

The sub-class whale can still invoke the method “walk” but **it shouldn’t**, and we must avoid it.

The way suggested by ISP is to create more ***client-specific interfaces\*** rather than ***one general-purpose interface\***. So, our code example becomes:

```python
from abc import ABC, abstractmethod

class Walker(ABC):
    @abstractmethod
    def walk() -> bool:
        return print('Can walk')
    
class Swimmer(ABC):
	@abstractmethod
    def swim() -> bool:
        return print('Can swim')
    
class Human(Walker, Swimmer):
    def walk():
        return print('Humans can walk')
    def swim():
        return print('Humans can swim')
    
class Whale(Swimmer):
    def swim():
        return print('Whales can swim')
    
if __name__ == "__main__":
    Human.walk()
    Human.swim()
    
    Whale.walk()
    Whale.swim()
```

Now, every sub-class inherits only what it needs, avoiding invoking an  out-of-context (wrong) sub-method. That might create an error hard to  catch.

This principle is closely connected with the other ones and specifically, it tells us to *keep the content of a subclass clean from elements of no use to that subclass.* This has the final aim to keep our classes clean and minimise mistakes.

# 5) The Dependency Inversion Principle (DIP)

> “*Abstractions should not depend on details. Details should depend on abstraction.  High-level modules should not depend on low-level modules. Both should  depend on abstractions”*

So, that abstractions (e.g., the interface, as seen above) should not be  dependent on low-level methods but both should depend on a third  interface.

To better explain this concept, I prefer to think of a sort of information flow.

Imagine that you have a program that takes in input a specific set of info (a  file, a format, etc) and you wrote a script to process it.
What would happen if that info were subject to changes? 
You would have to rewrite your script and adjust the new format. Losing the retro compatibility with the older files.

However, you could solve this by creating a third abstraction that takes the info as input and passes it to the others. 
This is basically what an API is also, used for.

![img](https://miro.medium.com/max/633/1*7rFi864XfIo2VGG9DCCF8g.png)

(Image by Author)

The interesting design concept of this principle is that it is the reverse approach to what we would normally do.

With the DIP in mind, we would start from the end of the project, in which  our code is independent of what takes in input and it is not susceptible to changes and out of our direct control.