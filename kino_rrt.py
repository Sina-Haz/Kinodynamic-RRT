# Tree data structure -> Holds states, each state can be a "Node" in the tree
import numpy as np
import mujoco
import mujoco_viewer as mjv
from typing import Union, Tuple

xml = 'nav1.xml'

class Node:
    def __init__(self, q,  qdot=np.array([0, 0]), parent=None) -> None:
        self.q = q
        self.qdot = qdot
        self.parent = parent
        # self.ctrl = None


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
        self.t = 0 # Time the simulation has run for
        model = mujoco.MjModel.from_xml_path(xml) # Mujoco model of our environment
        self.sim = mujoco.MjSim(model)
        self.xbds = xbounds # where we can sample from on x-axis (np.array w/ shape (2, ))
        self.ybds = ybounds # where we can sample from on y-axis (np.array w/ shape (2, ))
        self.ctrl_lim = ctrl_limits # Limits of our actuators, assuming they're uniform (np.array w/ shape (2, ))
        self.robot_id = mujoco.mj_name2id()

    def get_curr_state(self)->Node:
        '''
        This function indexes into the current mujoco data and returns a node containing the current configuration
        and velocity of the robot at this point in time
        '''
        return Node(self.sim.data.qpos, self.sim.data.qvel)

    def set_curr_state(self, x: Node)->bool:
        '''
        Index into the mujoco model and set it's current state according to the data in x and sets the simulation time to 0
        Returns boolean representing whether the state results in a collision or not
        True ->  No collision, False -> Collision
        '''
        # Set configuration and velocity
        self.sim.data.qpos[:] = x.q
        self.sim.data.qvel[:] = x.qdot

        self.sim.step() # step the simulation to update collision data

        self.sim.data.time = 0 # set the time to 0

        return self.sim.data.ncon == 0



    def sample_state(self)->Node:
        '''
        Uniformly samples: 
         - configuration within bounds  
         - qdot from standard normal
         - Returns a node with this data and uninitialized parent
        '''
        x, y = np.random.uniform(self.xbds[0],self.xbds[1]), np.random.uniform(self.ybds[0],self.ybds[1])
        qdot = np.random.normal(size=(2,))

        return Node(q=np.array([x,y]), qdot=qdot)
    

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

        sample_ctrls = [self.sample_ctrl() for i in range(n)]
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

        while self.sim.data.time < dt:
            self.sim.step()
            curr = self.get_curr_state()
            if self.data.ncon > 0: 
                collides = True
                break
        
        return (curr, collides, self.sim.data.time)
        




    
    def kRRT_PC():
        '''
        Sample controls completely at random and apply to x_near to extend the tree
        '''
        pass
    
    def kRRT_X():
        '''
        
        '''
