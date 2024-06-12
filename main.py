import time
import numpy as np
from tqdm import tqdm
from src.embedding import Embedding
from src.dynamical_system import DynamicalSystem
from src.forward_kinematics import ForwardKinematic
from utils.franka_parameters import joint_limits
from utils.messenger import Messenger


if __name__ == '__main__':

    fk = ForwardKinematic(
        urdf_file='franka_description/urdf/panda_no_gripper.urdf',
        components_per_link=1
        )

    # arbitrary target configuration
    config_attractor = np.array([-1.98, -0.34, -2.14, -2.74,  2.89, 0.80,  0.07])

    # placing an obstacle in the trajectory's way
    x = np.array([[-0.2], [-0.1], [2.6]])[np.newaxis, :]

    e = Embedding(dimension=fk.model.nq, x=x.repeat(1, 0), fk=fk, limits=joint_limits)

    K = 1 * np.eye(fk.model.nq)
    D = 1.5*np.eye(fk.model.nq)
    ds = DynamicalSystem(stiffness=K, dissipation=D, attractor=config_attractor, embedding=e, dt=0.01)

    # initial conditions
    q = np.array([0., 0., 0., -1.5, 0., 1.5, 0.])
    dq = np.zeros_like(q)

    # zmq streamer to the beautfiul-bullet simulator
    publisher = Messenger(port="5511")

    #iterate over the DS
    print(f"starting DS iteration. Target : {config_attractor}")
    for _ in tqdm(range(1000)):
        q, dq = ds(q, dq)
        publisher.publish(q.squeeze().tolist())
        time.sleep(1e-3)

    np.set_printoptions(precision=3, suppress=True)
    print(f"DS iteration finished. Final configuration : {q}")

