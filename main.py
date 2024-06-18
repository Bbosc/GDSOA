import json
import numpy as np
from tqdm import tqdm
from src.embedding import Embedding
from src.dynamical_system import DynamicalSystem
from src.forward_kinematics import ForwardKinematic
from utils.franka_parameters import joint_limits
from utils.messenger import Messenger


if __name__ == '__main__':

    with open('config/environment1.json') as file:
        config = json.load(file)

    fk = ForwardKinematic(
        urdf_file=config['urdf'],
        components_per_link=1
        )

    # arbitrary target configuration
    config_attractor = np.array([a * np.pi/180 for a in config['attractor']])
    # placing an obstacle in the trajectory's way
    x = np.array(config['obstacles'])

    e = Embedding(dimension=fk.model.nq, x=x, fk=fk, limits=joint_limits)

    K = 1 * np.eye(fk.model.nq)
    D = 1.5*np.eye(fk.model.nq)
    ds = DynamicalSystem(stiffness=K, dissipation=D, attractor=config_attractor, embedding=e, dt=0.01)

    # initial conditions
    q = np.array([a * np.pi/180 for a in config['initial_configuration']])
    dq = np.array(config['initial_velocities'])

    # zmq streamer to the beautfiul-bullet simulator
    publisher = Messenger(port="5511")

    #iterate over the DS
    print(f"starting DS iteration. Target : {config_attractor}")
    for _ in tqdm(range(1000)):
        q, dq = ds(q, dq)
        publisher.publish(q.squeeze().tolist())

    np.set_printoptions(precision=3, suppress=True)
    print(f"DS iteration finished. Final configuration : {q}")

