import json
import numpy as np
import time
from src.embedding import Embedding, Collision
from src.dynamical_system import DynamicalSystem
from src.forward_kinematics import ForwardKinematic
from utils.messenger import Client


if __name__ == '__main__':

    with open('config/environment2.json') as file:
        config = json.load(file)

    fk = ForwardKinematic(
        urdf_file=config['urdf'],
        gmm_configuration_file='config/gmm.json'
        )

    # arbitrary target configuration
    config_attractor = np.array([a * np.pi/180 for a in config['attractor']])
    # placing obstacles in the trajectory's way
    x = np.array(config['obstacles'])

    e = Embedding(
        embeddings=[Collision(x=x, fk=fk)]
    )

    K = 0.5 * np.eye(fk.model.nq)
    D = 1.5*np.eye(fk.model.nq)
    ds = DynamicalSystem(stiffness=K, dissipation=D, attractor=config_attractor, embedding=e, dt=0.001)

    # initial conditions
    q = np.array([a * np.pi/180 for a in config['initial_configuration']])
    dq = np.array(config['initial_velocities'])

    client = Client(port="5511")

    #iterate over the DS
    print(f"starting DS iteration. Target : {config_attractor}")
    for _ in range(13000):
        ddq = ds.compute_basic_switched_acceleration(q, dq, kappa=0.15)
        client.send_request(ddq.squeeze().tolist())
        q, dq = np.split(client.get_reply(), 2)
        time.sleep(1e-3)

    np.set_printoptions(precision=3, suppress=True)
    print(f"DS iteration finished. Final configuration : {q}")

