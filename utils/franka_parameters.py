from numpy import pi

# source : https://frankaemika.github.io/docs/control_parameters.html

DOF = 7


joint_limits = [
    {'lower': -2.8973, 'upper': 2.8973}, # joint 1
    {'lower': -1.7628, 'upper': 1.7628}, # joint 2
    {'lower': -2.8973, 'upper': 2.8973}, # joint 3
    {'lower': -3.0718, 'upper': -0.0698}, # joint 4
    {'lower': -2.8973, 'upper': 2.8973}, # joint 5
    {'lower': -0.0175, 'upper': 3.7525}, # joint 6
    {'lower': -2.8973, 'upper': 2.8973}, # joint 7
]

joint_limits_deg = list(
    map(
        lambda limit : {'lower': limit['lower']*180/pi, 'upper': limit['upper']*180/pi},
        joint_limits
    )
)

joint_acceleration_limits = [10 for _ in range(DOF)]