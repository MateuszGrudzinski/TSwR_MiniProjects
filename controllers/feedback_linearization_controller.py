import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp,0.1)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        kp = 30.5
        kd = 20
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot + kd*(q_r_dot-q_dot) + kp*(q_r-q)
        v_des = self.model.M(x)@v + self.model.C(x)@q_r_dot
        return v_des
