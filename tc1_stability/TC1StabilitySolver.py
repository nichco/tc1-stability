# run file for TC1 open-loop stability
import numpy as np
import csdl
from csdl import Model
import matplotlib.pyplot as plt
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import python_csdl_backend
import csdl_om

from tc1_stability.eig_long import Eig_Long
from tc1_stability.eig_lat import Eig_Lat
from tc1_stability.long import Long
from tc1_stability.lat import Lat


class TC1StabilityModel(Model):
    def initialize(self):
        # self.parameters.declare('A_long')
        # self.parameters.declare('A_lat')
        pass
    def define(self):
        size = 4
        # A_long = self.parameters['A_long']
        # A_lat = self.parameters['A_lat']

        A_matrix = self.declare_variable('A_matrix',shape=(12,12))

        # given states [u v w p q r phi theta psi x y z]
        # extract longitudinal fd states [u w q theta]
        A_long = self.create_output(name='A_long', shape=(4, 4))
        A_long[0,0] = A_matrix[0,0] # du/du
        A_long[0,1] = A_matrix[0,2] # du/dw
        A_long[0,2] = A_matrix[0,4] # du/dq
        A_long[0,3] = A_matrix[0,7] # du/dtheta
        A_long[1,0] = A_matrix[2,0] # dw/du
        A_long[1,1] = A_matrix[2,2] # dw/dw
        A_long[1,2] = A_matrix[2,4] # dw/dq
        A_long[1,3] = A_matrix[2,7] # dw/dtheta
        A_long[2,0] = A_matrix[4,0] # dq/du
        A_long[2,1] = A_matrix[4,2] # dq/dw
        A_long[2,2] = A_matrix[4,4] # dq/dq
        A_long[2,3] = A_matrix[4,7] # dq/dtheta
        A_long[3,0] = A_matrix[7,0] # dtheta/du
        A_long[3,1] = A_matrix[7,2] # dtheta/dw
        A_long[3,2] = A_matrix[7,4] # dtheta/dq
        A_long[3,3] = A_matrix[7,7] # dtheta/dtheta
        # given states [u v w p q r phi theta psi x y z]
        # extract lateral fd states [v p r phi]
        A_lat = self.create_output(name='A_lat', shape=(4, 4))
        A_lat[0,0] = A_matrix[1,1] # dv/dv
        A_lat[0,1] = A_matrix[1,3] # dv/dp
        A_lat[0,2] = A_matrix[1,5] # dv/dr
        A_lat[0,3] = A_matrix[1,6] # dv/dphi
        A_lat[1,0] = A_matrix[3,1] # dp/dv
        A_lat[1,1] = A_matrix[3,3] # dp/dp
        A_lat[1,2] = A_matrix[3,5] # dp/dr
        A_lat[1,3] = A_matrix[3,6] # dp/dphi
        A_lat[2,0] = A_matrix[5,1] # dr/dv
        A_lat[2,1] = A_matrix[5,3] # dr/dp
        A_lat[2,2] = A_matrix[5,5] # dr/dr
        A_lat[2,3] = A_matrix[5,6] # dr/dphi
        A_lat[3,0] = A_matrix[6,1] # dphi/dv
        A_lat[3,1] = A_matrix[6,3] # dphi/dp
        A_lat[3,2] = A_matrix[6,5] # dphi/dr
        A_lat[3,3] = A_matrix[6,6] # dphi/dphi
        
        # self.add(Eig_Long(size=size, val=A_long))
        self.add(Eig_Long(size=size))
        self.add(Long(size=size))
        
        # self.add(Eig_Lat(size=size, val=A_lat))
        self.add(Eig_Lat(size=size))
        self.add(Lat(size=size))


"""
A_long = np.array([[-3.10006462e-02,  1.35968193e-01,  4.47422538e+00, -3.21682896e+01],
                   [-3.60470947e-01, -2.25573434e+00,  2.01474173e+02,  6.05858233e-01],
                   [ 2.92702949e-03, -1.11344744e-01, -3.65414366e+00,  1.81446238e-10],
                   [ 0.00000000e+00,  0.00000000e+00,  9.99999941e-01,  0.00000000e+00]])

A_long_2 = np.array([[-9.91498896e-02, -2.71735986e-03, -1.56616139e+00, -2.83933297e-01],
                     [-3.25355791e-01, -8.91674306e-03,  5.74969557e+01, -1.62657457e+02],
                     [ 1.36965474e-05,  3.75369356e-07, -3.93070488e+00, -2.24950324e+01],
                     [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00]])

A_lat = np.array([[-0.2543,0.183,0,-1],
                  [0,0,1,0],
                  [-15.982,0,-8.402,2.193],
                  [4.495,0,-0.3498,-0.7605]])



# sim = python_csdl_backend.Simulator(dynamic_stability(size=size, A_long=A_long, A_lat=A_lat))
# sim = csdl_om.Simulator(TC1StabilityModel(A_long=A_long_2, A_lat=A_lat))
# sim = python_csdl_backend.Simulator(TC1StabilityModel(A_long=A_long_2, A_lat=A_lat))
sim = python_csdl_backend.Simulator(TC1StabilityModel())
sim.run()

print('----LONGITUDINAL----')
print('eigenvalues real (long):', sim['e_real_long'])
print('eigenvalues imag (long):', sim['e_imag_long'])
print('sp_wn   :', sim['sp_wn'])
print('ph_wn   :', sim['ph_wn'])
print('sp_z   :', sim['sp_z'])
print('ph_z   :', sim['ph_z'])
print('sp_t2   :', sim['sp_t2'])
print('ph_t2   :', sim['ph_t2'])

print('----LATERAL----')
print('eigenvalues real (lat):', sim['e_real_lat'])
print('eigenvalues imag (lat):', sim['e_imag_lat'])
print('dr_wn   :', sim['dr_wn'])
print('rr_wn   :', sim['rr_wn'])
print('ss_wn   :', sim['ss_wn'])
print('dr_z   :', sim['dr_z'])
print('rr_z   :', sim['rr_z'])
print('ss_z   :', sim['ss_z'])
print('dr_t2   :', sim['dr_t2'])
print('rr_t2   :', sim['rr_t2'])
print('ss_t2   :', sim['ss_t2'])
"""