# Machine-Learning
A basic machine learning program I wrote in Python

In this project, I simulate "Rockets" (squares) that travel around. They need to reach goals (in a certain order) and avoid obstacles. To do this, they have DNA, containing information on what direction they will travel each frame. Each generation, the best performing rockets are selected, copied and mutated randomly in order to improve the maximum score reached. This way they (hopefully) reach the final goal. 

For the scoring, I came up with a set of rules:
  - each frame, the score is updated for each of the rockets
  - the further away a rocket is from its current goal, the more points should be subtracted
  - the longer it takes a rocket to reach its current goal, the more points should be subtracted
  - (a lot of) points get added when the rocket reaches a goal
  
In order to force the rockets to go the best possible path, the evolution is split up into "stages". Every stage, the rockets try to reach one goal further. For the scoring, points only get added or subtracted until the current goal we are trying to reach this stage is reached. (So if we are on stage 0, reaching the 1st goal after the 0th goal does not add up to the score).

For the mutation:
While the stage number is less than the amount of goals we are trying to reach, only genes after the previous goal are mutated. After that, all genes are mutated again. This is why it is important to store at what frame a goal was reached. 

Later genes mutate more extreme, to allow better finding of the best route.

Each stage, the "randomness" of the mutation starts off at 1. (completely random) and if the score hasn't improved for so many generations, the randomness decreases. Once the randomness dips below a certain point, we move on to the next stage.
