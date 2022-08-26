# run file for TC1 open-loop stability

import numpy as np
import csdl
import matplotlib.pyplot as plt
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import python_csdl_backend
import csdl_om

from eig_long import Eig_Long
from eig_lat import Eig_Lat
from long import Long
from lat import Lat


class dynamic_stability(csdl.Model):
    def initialize(self):
        self.parameters.declare('size')
        self.parameters.declare('A_long')
        self.parameters.declare('A_lat')
    def define(self):
        size = self.parameters['size']
        
        A_long = self.parameters['A_long']
        A_lat = self.parameters['A_lat']
        
        self.add(Eig_Long(size=size, val=A_long))
        self.add(Long(size=size))
        
        self.add(Eig_Lat(size=size, val=A_lat))
        self.add(Lat(size=size))

size = 4

A_long = np.array([[-3.10006462e-02,  1.35968193e-01,  4.47422538e+00, -3.21682896e+01],
                   [-3.60470947e-01, -2.25573434e+00,  2.01474173e+02,  6.05858233e-01],
                   [ 2.92702949e-03, -1.11344744e-01, -3.65414366e+00,  1.81446238e-10],
                   [ 0.00000000e+00,  0.00000000e+00,  9.99999941e-01,  0.00000000e+00]])
A_lat = np.array([[-0.2543,0.183,0,-1],
                  [0,0,1,0],
                  [-15.982,0,-8.402,2.193],
                  [4.495,0,-0.3498,-0.7605]])


#sim = python_csdl_backend.Simulator(dynamic_stability(size=size, A_long=A_long, A_lat=A_lat))
sim = csdl_om.Simulator(dynamic_stability(size=size, A_long=A_long, A_lat=A_lat))
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