import mujoco
import mujoco_viewer as mjv


xml = 'nav1.xml'

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(xml)

# Create the simulation data structure
data = mujoco.MjData(model)

# Create a viewer to render the simulation
viewer = mjv.MujocoViewer(model, data)

while True:
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()