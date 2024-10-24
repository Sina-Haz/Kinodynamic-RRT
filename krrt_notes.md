An extension of regular [[Rapidly-Exploring Random Tree|RRT]] for [[Kinodynamic Planning]] with motion constraints

#### Simulation Function 
Need to define a simulation function to perform forward planning $$x_{next} = h(x, u)$$
 - Use this to to apply sampling based methods to the control space
 - Tree growing techniques better in this case --> why we use RRT

#### Main Idea
 - Maintain a tree $T$ of states connected by feasible paths rooted from the start
 - Grow the tree by sampling a control from an existing state and then applying the simulation function 
 - Terminates when a state is found in the goal region

### Algorithm
```
Kinodynamic-RRT:
	for i in range(N):
		x_rand = sample()
		x_near = nearest(T, x_rand)
		u_e = Choose-Control(x_near, x_rand)
		x_e = Simulate(x_near, u_e)
		if not collision(path(x_near, x_e)):
			add edge x_near -> x_e to T
		if x_e in Goal:
			return path(x_0, x_e)
	return No Path
```

__Metric Choice__:
 - Important to find a feasible solution
 - Euclidean distance may not always be a good idea
	 - For example car, sideways euclidean distance (parallel parking) actually involves a lot of going forwards and backwards to accomplish
- Can use [[Linear Quadratic Regulator|LQR]] for distance metric somehow??

**Choose-Control**:
 - One approach is to choose control that moves from x_near to x_rand
$$u_e = \arg \min_{u \in U} d(x_{rand}, \text{Simulate}(x_{near}, u))$$
 - This can be done by sampling a few random controls and choosing the one that gets closest to $x_{rand}$
	 - Makes rapid progress
	 - Can get stuck though
- In order for KRRT to be probabilistically complete though, it must sample controls at random

**Goal Region instead of Goal State**: 
 - Chance of arriving to an exact single state is 0
 - Hence we use a goal region
 - If [[Steering Function]] was available could replace $x_e \in G$ with test to see if can connect to goal state
