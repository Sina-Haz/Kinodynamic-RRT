# Tree data structure -> Holds states, each state can be a "Node" in the tree
import numpy as np
import mujoco
import mujoco_viewer as mjv
from typing import Union, Tuple
import time

xml = 'nav1.xml'
T_max = 30

class Node:
    def __init__(self, q,  qdot=np.array([0, 0]), parent=None) -> None:
        self.q = q
        self.qdot = qdot
        self.parent = parent
        self.ctrl = None

    
    def __repr__(self) -> str:
        return f'Node(q = {self.q}, qdot = {self.qdot}, ctrl = {self.ctrl})'


def weighted_euclidean_distance(x1: Node, x2: Node, pw: float = 1, vw: float = 0.1) -> float:
    '''
    Computes weighted euclidean distance between nodes x1 and x2 with pw being the position weight and vw the velocity weight
    Since for this application we mainly want to reach goal region, don't care as much about velocity and weigh position more
    '''
    pos_diff = np.linalg.norm(x2.q - x1.q)
    v_diff = np.linalg.norm(x2.qdot - x1.qdot)
    weighted_dist = np.sqrt(pw*pos_diff**2 + vw*v_diff**2)
    return weighted_dist



def ctrl_effort_distance(x1: Node, x2: Node) -> float:
    raise NotImplementedError



def sample_state(bounds: np.array)->Node:
    '''
    Uniformly samples: 
        - configuration within bounds  
        - qdot from standard normal
        - Returns a node with this data and uninitialized parent
    '''
    # Bounds have the shape (q, 2) -> each row is for a q_i has lower and upper bound for q_i
    q = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])

    # Return qdot from a standard normal population with same shape as q
    qdot = np.random.normal(size=q.shape)

    return Node(q=q, qdot=qdot)


def sample_non_colliding(sampler_fn, collision_checker, sample_bounds):
    '''
    A generic function that takes in a sampler and collision checker as function pointers and continues sampling with
    The sampler until it gets to a non-colliding state.

    sampler_fn should only need bounds as argument (everything else either keyword argument or not provided here)

    collision checker should take in output of sampler function and return True if no collision, False if collision
    '''
    while True:
        sample = sampler_fn(sample_bounds)
        # If no collision from sample -> return this
        if collision_checker(sample):
            break
    
    return sample


class KRRT:
    def __init__(self,
                 start: np.array,
                 goal: np.array,
                 xbounds: np.array,
                 ybounds: np.array,
                 ctrl_limits: np.array
                ) -> None:
        self.Tree = [Node(start)] # Tree currently stored as list, later may implement k-d tree for faster check
        self.goal = goal # Stores goal config
        model = mujoco.MjModel.from_xml_path(xml) # Mujoco model of our environment
        self.model = model
        self.data = mujoco.MjData(model)
        self.xbds = xbounds # where we can sample from on x-axis (np.array w/ shape (2, ))
        self.ybds = ybounds # where we can sample from on y-axis (np.array w/ shape (2, ))
        self.ctrl_lim = ctrl_limits # Limits of our actuators, assuming they're uniform (np.array w/ shape (2, ))
        # self.robot_id = mujoco.mj_name2id()

    def get_curr_state(self)->Node:
        '''
        This function indexes into the current mujoco data and returns a node containing the current configuration
        and velocity of the robot at this point in time
        '''
        return Node(self.data.qpos, self.data.qvel)

    def set_curr_state(self, x: Node)->bool:
        '''
        Index into the mujoco model and set it's current state according to the data in x and sets the simulation time to 0
        Returns boolean representing whether the state results in a collision or not
        True ->  No collision, False -> Collision
        '''
        # Set configuration and velocity
        self.data.qpos[:] = x.q
        self.data.qvel[:] = x.qdot

        mujoco.mj_step(self.model, self.data) # step the simulation to update collision data

        self.data.time = 0 # Set the model time to 0

        return self.data.ncon == 0
    

    def nearest_neighbor(self, x: Node, distance_metric = weighted_euclidean_distance) -> Node:
        '''
        Takes in a node and distance metric function and computes nearest Tree Node to the input and returns it
        '''
        min_dist = float('inf')
        nearest = self.Tree[0] # Use this instead of None so there's no chance of returning something that's not a Node

        for n in self.Tree:
            dist = distance_metric(n, x)
            if min_dist > dist:
                min_dist = dist
                nearest = n
        return nearest


    def sample_ctrl(self)->np.array:
        '''
        Uniformly samples an x, y control within the defined control limits. Returns as a vector of shape (2,)
        '''
        return np.random.uniform(self.ctrl_lim[0], self.ctrl_lim[1], size=(2,))
    

    def sample_best_ctrl(self, 
                         x1: Node,
                         x2: Node,
                         n=10,
                         dt=0.1,
                         distance_metric=weighted_euclidean_distance
                        ) -> Union[Tuple[np.ndarray, Node], Tuple[None, Node]]:
        '''
        Takes in 2 nodes and generates n sample controls. 
         - Simulates each, starting at x1, for dt duration
         - Returns the sampled control that generates state with minimum distance from x2

         Either returns the best control and x_e or None and x1 if all of sampled the controls resulted in a collision
        '''

        sample_ctrls = [self.sample_ctrl() for _ in range(n)]
        min_dist = float('inf')
        best_ctrl, x_e = None, x1

        for ctrl in sample_ctrls:
            x_e, collides, t = self.simulate(x1, ctrl, dt)
            if not collides:
                dist = distance_metric(x_e, x2)
                if dist < min_dist:
                    min_dist = dist
                    best_ctrl = ctrl
        return best_ctrl, x_e



    def simulate(self, state, ctrl, dt=0.1) -> Tuple[Node, bool, float]:
        '''
        Takes in a start state and control. It applies the control to the state for dt duration.
        Returns the final state, whether or not there was a collision, and how much time passed
        '''
        collides = not self.set_curr_state(state)

        if collides: return (state, True, 0) # Don't even bother simulating

        while self.data.time < dt:
            mujoco.mj_step(self.model, self.data)
            curr = self.get_curr_state()
            if self.data.ncon > 0: 
                collides = True
                break
        
        return (curr, collides, self.data.time)
        

    def in_goal(self, state: Node):
        #id = self.model.site_name2id('target4')
        goal_pos = self.model.site("target4").pos
        goal_size = self.model.site("target4").size
        x_min = goal_pos[0] - (goal_size[0]/2)
        x_max = goal_pos[0] + (goal_size[0]/2)
        y_min = goal_pos[1] - (goal_size[1]/2)
        y_max = goal_pos[1] + (goal_size[1]/2)
        if (x_min < state.q[0] < x_max):
            if (y_min < state.q[1] < y_max):
                return True
       
        return False



    
    def kRRT(self):
        '''
        Runs the kRRT algorithm using the data and methods within the class.
        Goes as follows:
        while time < T_max:
            x_rand = sample()
            x_near = Nearest(Tree, x_rand)
            u_e = Choose-Control(x_near, x_rand)
            x_e = Simulate(x_near, u_e)
            if not collision(path(x_near, x_e)):
			    add edge x_near -> x_e to T
		    if x_e in Goal:
			    return path(x_0, x_e)
	    return No Path
        '''
        curr = 0
        while curr < T_max: # Small for loop to check functionality
            start = time.time()
            x_rand = sample_non_colliding(sampler_fn=sample_state,
                                          collision_checker=self.set_curr_state,
                                          sample_bounds=np.array([self.xbds, self.ybds]))
            # This is a node within our tree so unless it's the start its guaranteed to have a parent
            x_near = self.nearest_neighbor(x_rand)

            u, x_e = self.sample_best_ctrl(x_near, x_rand)
            if u is not None: #i.e. if there was a control that didn't result in a collision
                x_e.parent = x_rand
                x_e.ctrl = u
                self.Tree.append(x_e)
            
                if self.in_goal(x_e):
                    print(curr + (time.time() - start), len(self.Tree))
                    return True # TODO: replace this with get_path function once Himani implements
            end = time.time()
            curr += (end - start)
        print(curr)
        return False


# Remember to treat this as a 2D problem since we have no joint on the z-axis
# Currently the goal attribute not used, maybe change later
test1 = KRRT(start=np.array([0, 0]), goal=None, xbounds=[-.2, 1.1], ybounds=[-.36, .36], ctrl_limits=[-.5, .5])
print(test1.kRRT())
# print(test1.Tree)