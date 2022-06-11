import csdl
import numpy as np

class Long(csdl.Model):

    def initialize(self):
        self.parameters.declare('size')
    def define(self):
        size = self.parameters['size']
        
        e_real = self.declare_variable('e_real_long', shape=(1,size))
        e_imag = self.declare_variable('e_imag_long', shape=(1,size))
        
        # short period eigenvalue pair
        sp_e_real = e_real[0,2]
        sp_e_imag = e_imag[0,2]
        self.register_output('sp_e_real', sp_e_real)
        self.register_output('sp_e_imag', sp_e_imag)

        # get phugoid eigenvalue pair
        ph_e_real = e_real[0,0]
        ph_e_imag = e_imag[0,0]
        self.register_output('ph_e_real', ph_e_real)
        self.register_output('ph_e_imag', ph_e_imag)

        
        # compute short period natural frequency
        sp_wn = (sp_e_real**2 + sp_e_imag**2)**0.5
        self.register_output('sp_wn', sp_wn)
        # compute phugoid natural frequency
        ph_wn = (ph_e_real**2 + ph_e_imag**2)**0.5
        self.register_output('ph_wn', ph_wn)
        
        # compute short period damping ratio
        sp_z = -1*sp_e_real/sp_wn
        self.register_output('sp_z', sp_z)
        # compute phugoid damping ratio
        ph_z = -1*ph_e_real/ph_wn
        self.register_output('ph_z', ph_z)
        
        # compute short period time to double
        sp_z_abs = (sp_z**2)**0.5
        sp_t2 = np.log(2)/(sp_z_abs*sp_wn)
        self.register_output('sp_t2', sp_t2)
        # compute phugoid time to double
        ph_z_abs = (ph_z**2)**0.5
        ph_t2 = np.log(2)/(ph_z_abs*ph_wn)
        self.register_output('ph_t2', ph_t2)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
