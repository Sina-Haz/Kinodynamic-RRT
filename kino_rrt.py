# Tree data structure -> Holds states, each state can be a "Node" in the tree
import numpy as np
import mujoco


xml = 'nav1.xml'

class Node:
    def __init__(self, q,  qdot, parent) -> None:
        self.q = q
        self.qdot = qdot
        self.parent = parent
        self.ctrl = None


class KRRT:
    def __init__(self, start: Node, goal: Node) -> None:
        self.Tree = [start]
        self.goal = goal
        self.t = 0
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

    def sample_cfg(self)->np.array:
        pass

    def nearest_neighbor(self, x: Node)->Node:
        pass

    def sample_ctrl(self)->np.array:
        pass

    def simulate_ctrl(self, nearest_idx, ctrl)->bool:
        pass

    
