# Tree data structure -> Holds states, each state can be a "Node" in the tree
import numpy as np
import mujoco
import mujoco_viewer as mjv
import matplotlib.pyplot as plt
from typing import Union, Tuple
import time

xml = 'nav1.xml'
T_max = 30
np.set_printoptions(precision=3)

class Node:
    def __init__(self, q: np.array,  qdot:np.array = np.array([0, 0]), parent=None) -> None:
        self.q = q
        self.qdot = qdot
        self.parent = parent
        self.ctrl = None

    
    def __repr__(self) -> str:
        return f'Node(q = {self.q}, qdot = {self.qdot}, ctrl = {self.ctrl})'

    def __eq__(self, other):
        return np.array_equal(self.q, other.q) and np.array_equal(self.qdot, other.qdot) and (self.parent == other.parent) and np.array_equal(self.ctrl, other.ctrl)


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
        return Node(np.copy(self.data.qpos), np.copy(self.data.qvel))

    def set_curr_state(self, x: Node)->bool:
        '''
        Index into the mujoco model and set it's current state according to the data in x and sets the simulation time to 0
        Returns boolean representing whether the state results in a collision or not
        True ->  No collision, False -> Collision
        '''
        # Set configuration and velocity
        self.data.qpos[:] = x.q
        self.data.qvel[:] = x.qdot

        mujoco.mj_forward(self.model, self.data) # step the simulation to update collision data

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

         Either returns the best control and x_e or None and None if all of sampled the controls resulted in a collision
        '''

        sample_ctrls = [self.sample_ctrl() for _ in range(n)]
        min_dist = float('inf')
        best_ctrl, x_e = None, None

        for ctrl in sample_ctrls:
            x_e, collides, t = self.simulate(x1, ctrl, dt)
            if x_e is not None and not collides:
                dist = distance_metric(x_e, x2)
                if dist < min_dist:
                    min_dist = dist
                    best_ctrl = np.array(ctrl)
        return best_ctrl, x_e



    def simulate(self, state, ctrl, dt=0.1) -> Tuple[Node, bool]:
        '''
        Takes in a start state and control. It applies the control to the state for dt duration.
        Returns the final state, whether or not there was a collision, and how much time passed
        '''
        # Reset the simulation state
        self.set_curr_state(state)
        self.data.ctrl[:] = ctrl  # Explicitly set all control values
        self.data.time = 0
        
        # Simulation parameters
        timestep = self.model.opt.timestep  # Get the model's timestep
        num_steps = int(dt / timestep)  # Calculate number of steps needed
        
        collides = False
        
        # Run simulation for specified duration
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
            
            if self.data.ncon > 0:
                collides = True
                break
        
        # Get final state
        curr = self.get_curr_state()
        
        return curr, collides
        

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
        currT = 0
        while currT < T_max: # Small for loop to check functionality
            start = time.time()

            x_rand = sample_non_colliding(sampler_fn=sample_state,
                                          collision_checker=self.set_curr_state,
                                          sample_bounds=np.array([self.xbds, self.ybds]))
            
            # This is a node within our tree so unless it's the start its guaranteed to have a parent
            x_near = self.nearest_neighbor(x_rand)

            u = self.sample_ctrl()
            x_e, collides = self.simulate(x_near, u, dt=0.2)
            if collides: 
                currT += (time.time() - start)
                continue # Don't do all this if it collides

            x_e.parent = x_near
            x_e.ctrl = u
            self.Tree.append(x_e)

            if self.in_goal(x_e):
                print(currT + (time.time() - start), len(self.Tree))
                self.visualize_tree()
                return self.recreate_path() # TODO: replace this with get_path function once Himani implements
            end = time.time()
            currT += (end - start)
        print(currT)
        return None
    
    def recreate_path(self):
        '''
        x_final
        while curr.parent:
            add to list (curr)
            curr = curr.parent
        '''
        nodes = self.Tree.copy()
        nodes.reverse()
        path = [nodes[0]]
        parent = nodes[0].parent
        for node in nodes[:-1]: 
            #equality is off
            if (node == parent):
                path.append(node)
                parent = node.parent
        path.append(nodes[-1])
        path.reverse()
        return path
    
    def visualize_tree(self) :
        fig, ax = plt.subplots()
        ax.plot(self.Tree[0].q[0], self.Tree[0].q[1], 'bo', label = 'Start')
        ax.set_xlim(self.xbds)
        ax.set_ylim(self.ybds)
        
        #maybe graph obstacle?
        
        for node in self.Tree[1:]:
            ax.plot([node.q[0], node.parent.q[0]], [node.q[1], node.parent.q[1]], "-b")
        plt.show()




# # Remember to treat this as a 2D problem since we have no joint on the z-axis
# # Currently the goal attribute not used, maybe change later
test1 = KRRT(start=np.array([0, 0]), goal=None, xbounds=[-.2, 1.1], ybounds=[-.36, .36], ctrl_limits=[-1.,1.])
print(test1.kRRT())

# # print(test1.kRRT_PC())
# # print(test1.Tree)


# s1 = Node(q=np.array([.1,0]))
# u = np.array([0.5, 0.5])
# print(u)
# test1.data.ctrl = u
# dt = 0.5
# new, _ = test1.simulate(s1, u, dt=dt)
# # test1.set_curr_state(s1)
# # while test1.data.time < dt:
# #     mujoco.mj_step(test1.model, test1.data)
# #     new = test1.get_curr_state()

# print(np.allclose(s1.q, new.q))

