
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, state_dim, process_noise_cov, measurement_noise_cov, initial_state, initial_covariance):
        self.state_dim = state_dim
        self.Q = process_noise_cov
        self.R = measurement_noise_cov
        self.state = initial_state
        self.P = initial_covariance

    def predict(self, state_transition_func, jacobian_func, control_input=None):
        self.state = state_transition_func(self.state, control_input)
        F = jacobian_func(self.state, control_input)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement, measurement_func, jacobian_func):
        H = jacobian_func(self.state)
        y = measurement - measurement_func(self.state)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

def state_transition_func(state, control_input):
    return state

def measurement_func(state):
    return state[:2]
