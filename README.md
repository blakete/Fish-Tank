# Fish-Tank
Evolutionary cell simulation.<br><br>

## Cell Intelligence
1. Cell calculates fields of view (FOV)
  1. Cell can "see" foods and walls North, South, East, West within 75 pixels.
  2. This FOV is represented as a (1,8) vector. Indices 0-3 represent and 4-7 represent N,S,E,W for foods and walls respectively.
2. Cell FOV propogated through cell neural network
3. Neural network output (1,2) interpreted to control cell movement.
  * |Output[0]| < 1 --> 0 pixel x-axis movement
  * Output[0] > 1 --> 1 pixel x-axis movement
  * Output[0] < -1 --> -1 pixel x-axis movement
  * |Output[1]| < 1 --> 0 pixel y-axis movement
  * Output[1] > 1 --> 1 pixel y-axis movement
  * Output[1] < -1 --> -1 pixel y-axis movement

## Cell Environment & Lifecycle
* C cells start off with randomly initialized neural networks. F foods are randomly placed in the environment. 
* For each time step, each cell calculates its FOV and passes it through its neural network to determine its next movement.
* If a cell collides with a food, it "eats" it, and the cell gets +1 fitness point added to its fitness score.
* After X time steps, the generation ends. 
* The C cells are sorted by fitness score. TODO

