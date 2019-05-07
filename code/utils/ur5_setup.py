import numpy as np

setup = {
    'host': '192.168.1.100', # put UR5 Controller address here (192.168.2.152) (10.44.60.122)
    'end_effector_low': np.array([-0.2, -0.3, 0.5]),
    'end_effector_high': np.array([0.2, 0.4, 1.0]),
    'angles_low':np.pi/180 * np.array(
        [ 60,
        -180,#-180
        -120,
        -50,
         50,
         50
        ]
     ),
     'angles_high':np.pi/180 * np.array(
         [ 90,
          -60,
          130,
           25,
          120,
          175
         ]
     ),
     'speed_max': 0.3,   # maximum joint speed magnitude using speedj
     'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
     'reset_speed_limit': 0.5,
     'q_ref': np.array([ 1.58724391, -2.4, 1.5, -0.71790582, 1.63685572, 1.00910473]),
     'box_bound_buffer': 0.001,
     'angle_bound_buffer': 0.001,
     'ik_params':
      (
         0.089159, # d1
        -0.42500, # a2
        -0.39225, # a3
         0.10915,  # d4
         0.09465,  # d5
         0.0823    # d6
      )
}