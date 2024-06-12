import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        m1 = ManiuplatorModel(Tp,m3=0.1,r3=0.05)
        # II:  m3=0.01, r3=0.01
        m2 = ManiuplatorModel(Tp,m3=0.01,r3=0.01)
        # III: m3=1.0,  r3=0.3
        m3 = ManiuplatorModel(Tp,m3=1.0,r3=0.3)
        self.models = [m1, m2, m3]
        self.Tp = Tp
        self.prev_x = np.zeros(4)
        self.prev_u = np.zeros(2)

        self.i = 0

    def choose_model(self, x):
        q = x[:2]
        q_dot = x[2:]
        model_outputs = [model.x_dot(self.prev_x, self.prev_u) * self.Tp + self.prev_x.reshape(4,1) for model in self.models]
        errors = [np.sum(np.abs(x.reshape(4,1) - model_output)) for model_output in model_outputs]
        return np.argmin(errors)



    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        print("Wybrano Model: ", self.i)
        self.i = self.choose_model(x)
        kp = 8
        kd = 5
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot + kd * (q_r_dot - q_dot) + kp * (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prev_u = u
        self.prev_x = x
        return u
