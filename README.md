## Introduction & Objective
Inspired by how human brain matures much slower than other animals, so I thought perhaps there is some sort of 
benefit in progressive development/training. In the project, I will use residual network as the base network
and add additional residual blocks on top of the existing layers as the training goes on. The results will be 
compared with a residual network with all layers available at the very beginning.

#### Approach
Since residual blocks add to the initial input x instead of changing it directly, we can assume
that there are additional layers with weights of zero and biases of zero, here I will refer it as
a zero block 
(because these layers don't actually do anything e.g x' = x + wx + b = x + 0w + 0, x' = x).
After some training iterations we will introduce more and more block onto base/core network and see its
affect on the training accuracy
#### Potential Benefits 

1. Faster training time - Since the zero blocks can be removed from the beginning and only add them,
when they are needed, thus requiring
less calculations. (I have removed this feature from the program, but you can still 
use command line arguments to simulate it)
1. More versatile to new training data, perhaps it will be more accepting for new and slightly different
data. (Not confirmed)
1. Better fit/more general (Not confirmed, its accuracy is around the same as a network will all layers available
from the start in my test runs)
1. Adaptive complexity: perhaps for devices that are not powerful enough, they can run the core network instead
of the whole thing to decrease computation time (Not confirmed)

#### Running Instructions
Read the declaration of the flags in main.py\
`python3 main.py --{flag name}={some value} --{flag name}={some value}...`