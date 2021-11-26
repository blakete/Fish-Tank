# Fish Tank
Digital fish tank for your living room. Watch them as they evolve to find food!<br>

## Cell Intelligence
* Cell calculates fields of view (FOV)
  * Cell can "see" foods and walls North, South, East, West within 75 pixels.
  * This FOV is represented as a (1,8) vector. Indices 0-3 represent and 4-7 represent N,S,E,W for foods and walls respectively.
* Cell FOV propogated through cell neural network
* Neural network output (1,2) interpreted to control cell movement.
  * |Output[0]| < 1 --> 0 pixel x-axis movement
  * Output[0] > 1 --> 1 pixel x-axis movement
  * Output[0] < -1 --> -1 pixel x-axis movement
  * |Output[1]| < 1 --> 0 pixel y-axis movement
  * Output[1] > 1 --> 1 pixel y-axis movement
  * Output[1] < -1 --> -1 pixel y-axis movement

## Cell Environment & Lifecycle
* C cells start off with randomly initialized neural networks. F foods are randomly placed in the environment. 
* For each time step, each cell calculates its FOV and passes it through its neural network to determine its next movement.
* For each time step, each cell decreases in fitness at a constant rate. When a cell fitness hits 0, it "dies" and is destroyed.
* If a cell collides with a food, it "eats" it, and the cell gets +1 fitness point added to its fitness score.
* If a cell eats 3 foods, it is able to reproduce. If it is within 75 pixels of another cell ready to reproduce, a child cell will be created. The child cell will have a 50-50 chance of inheriting each of its parents' neural network weights. A small mutation will also be added to the child cell. 
*  

## References
* https://www.youtube.com/watch?v=qv6UVOQ0F44&ab_channel=SethBling
* http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
