import mujoco
import mujoco_viewer as mjv


xml = 'nav1.xml'

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(xml)

# Create the simulation data structure
data = mujoco.MjData(model)

# Create a viewer to render the simulation
viewer = mjv.MujocoViewer(model, data)

jx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint_x')
jy_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint_y')

print(f'Joint indices: {jx_id, jy_id}')

while True:
    mujoco.mj_step(model, data)
    viewer.render()
    # if i % 20 == 0: print(f'Full world configuration: {data.qpos}, Full world velocities: {data.qvel}')

viewer.close()
