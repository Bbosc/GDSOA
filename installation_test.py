import json
import numpy as np
from src.embedding import Embedding, Collision
from src.dynamical_system import DynamicalSystem
from src.forward_kinematics import ForwardKinematic
from utils.messenger import Client


if __name__ == '__main__':

    with open('config/environment2.json') as file:
        config = json.load(file)

    fk = ForwardKinematic(
        urdf_file=config['urdf'],
        gmm_configuration_file='config/gmm_unit.json'
        )

    # arbitrary target configuration
    config_attractor = np.array([a * np.pi/180 for a in config['attractor']])
    # placing obstacles in the trajectory's way
    x = np.array(config['obstacles'])

    e = Embedding(
        embeddings=[Collision(x=x, fk=fk)]
    )

    K = 0.5 * np.eye(fk.model.nq)
    D = 1.5 * np.eye(fk.model.nq)
    ds = DynamicalSystem(
        stiffness=K, dissipation=D, attractor=config_attractor, embedding=e
    )

    # initial conditions
    q = np.array([a * np.pi/180 for a in config['initial_configuration']])
    dq = np.array(config['initial_velocities'])

    client = Client(port="5511")

    q, dq = ds(q, dq, kappa=0.2)

