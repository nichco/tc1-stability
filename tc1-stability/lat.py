import csdl
import numpy as np
       

class Lat(csdl.Model):

    def initialize(self):
        self.parameters.declare('size')
    def define(self):
        size = self.parameters['size']
        
        e_real = self.declare_variable('e_real_lat', shape=(1,size))
        e_imag = self.declare_variable('e_imag_lat', shape=(1,size))
        # order: dr, dr, rr, ss
        
        # get dutch roll eigenvalue pair
        dr_e_real = e_real[0,0]
        dr_e_imag = e_imag[0,0]
        self.register_output('dr_e_real', dr_e_real)
        self.register_output('dr_e_imag', dr_e_imag)
        
        # get roll eigenvalue pair
        rr_e_real = e_real[0,2]
        rr_e_imag = e_imag[0,2]
        self.register_output('rr_e_real', rr_e_real)
        self.register_output('rr_e_imag', rr_e_imag)
        
        # get spiral eigenvalue pair
        ss_e_real = e_real[0,3]
        ss_e_imag = e_imag[0,3]
        self.register_output('ss_e_real', ss_e_real)
        self.register_output('ss_e_imag', ss_e_imag)

        # compute dutch roll natural frequency
        dr_wn = (dr_e_real**2 + dr_e_imag**2)**0.5
        self.register_output('dr_wn', dr_wn)
        # compute roll natural frequency
        rr_wn = (rr_e_real**2 + rr_e_imag**2)**0.5
        self.register_output('rr_wn', rr_wn)
        # compute spiral natural frequency
        ss_wn = (ss_e_real**2 + ss_e_imag**2)**0.5
        self.register_output('ss_wn', ss_wn)
        
        # compute dutch roll damping ratio
        dr_z = -1*dr_e_real/dr_wn
        self.register_output('dr_z', dr_z)
        # compute roll damping ratio
        rr_z = -1*rr_e_real/rr_wn
        self.register_output('rr_z', rr_z)
        # compute spiral damping ratio
        ss_z = -1*ss_e_real/ss_wn
        self.register_output('ss_z', ss_z)
        
        # compute dutch roll time to double
        dr_z_abs = (dr_z**2)**0.5
        dr_t2 = np.log(2)/(dr_z_abs*dr_wn)
        self.register_output('dr_t2', dr_t2)
        # compute roll time to double
        rr_z_abs = (rr_z**2)**0.5
        rr_t2 = np.log(2)/(rr_z_abs*rr_wn)
        self.register_output('rr_t2', rr_t2)
        # compute spiral time to double
        ss_z_abs = (ss_z**2)**0.5
        ss_t2 = np.log(2)/(ss_z_abs*ss_wn)
        self.register_output('ss_t2', ss_t2)
        
        
        
        
        
        
        
        
        
        