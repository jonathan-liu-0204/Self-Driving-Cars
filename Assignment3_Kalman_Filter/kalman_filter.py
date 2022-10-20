import numpy as np

class KalmanFilter():
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw])
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        # Error matrix
        self.P = np.identity(3) * 1
        # Observation matrix
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])
                
        # State transition error covariance
        # self.Q = np.eye(3)*0.0009
        self.Q = np.array([[0.08, 0.0, 0.0],
                           [0.0, 0.009, 0.0], 
                           [0.0, 0.0, 1.0]])

        # Measurement error
        #self.R = np.eye(2)*0.08
        self.R = np.array([[0.5, 0.00],
                           [0.00, 0.7]])

    def predict(self, u):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        # raise NotImplementedError

    def update(self, z):
        
        S = self.R + np.dot(np.dot(self.H, self.P), self.H.T)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        I = np.eye(3)
        # self.P = np.dot(I - np.dot(K, self.H), self.P)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x, self.P
        # raise NotImplementedError
