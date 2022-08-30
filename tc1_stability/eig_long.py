import csdl
import csdl_om
import numpy as np


class Eig_Long(csdl.Model):

    def initialize(self):
        # size and value of A
        self.parameters.declare('size')
        # self.parameters.declare('val')

    def define(self):
        # size and value of A
        size = self.parameters['size']
        shape = (size, size)
        # val = self.parameters['val']

        # Create A matrix
        # A_long = self.create_input('A_long', val=val, shape=shape)
        A_long = self.declare_variable('A_long', shape=(size,size))

        # custom operation insertion
        e_r, e_i = csdl.custom(A_long, op=EigExplicit(size=size))

        # eigenvalues as output
        self.register_output('e_real_long', e_r)
        self.register_output('e_imag_long', e_i)


class EigExplicit(csdl.CustomExplicitOperation):

    def initialize(self):
        # size of A
        self.parameters.declare('size')

    def define(self):
        # size of A
        size = self.parameters['size']
        shape = (size, size)

        # Input: Matrix
        self.add_input('A_long', shape=shape)

        # Output: Eigenvalues
        self.add_output('e_real_long', shape=(1,size))
        self.add_output('e_imag_long', shape=(1,size))

        self.declare_derivatives('e_real_long', 'A_long')
        self.declare_derivatives('e_imag_long', 'A_long')

    def compute(self, inputs, outputs):

        # Numpy eigenvalues
        w, v = np.linalg.eig(inputs['A_long'])
        
        # longitudinal eigenvalue classification
        eig_vals_mag = np.absolute(np.real(w))
        eig_val_mag_max = np.amax(eig_vals_mag)
        locSP  = np.where(eig_vals_mag == eig_val_mag_max)
        locPh = np.where(eig_vals_mag < eig_val_mag_max)
        
        sp_eig = w[locSP[0][0]]
        ph_eig = w[locPh[0][0]]
        
        e_real_sp = np.real(sp_eig)
        e_imag_sp = np.imag(sp_eig)
        
        e_real_ph = np.real(ph_eig)
        e_imag_ph = np.imag(ph_eig)
        
        # phugoid eigenvalues are first, followed by short period eigenvalues
        e_real = np.array([[e_real_ph, e_real_ph, e_real_sp, e_real_sp]])
        e_imag = np.array([[e_imag_ph, e_imag_ph, e_imag_sp, e_imag_sp]])
        
        outputs['e_real_long'] = 1*e_real
        outputs['e_imag_long'] = 1*e_imag

    def compute_derivatives(self, inputs, derivatives):
        size = self.parameters['size']
        shape = (size, size)

        # v are the eigenvectors in each columns
        w, v = np.linalg.eig(inputs['A_long'])

        # v inverse transpose
        v_inv_T = (np.linalg.inv(v)).T

        # preallocate Jacobian: n outputs, n^2 inputs
        temp_r = np.zeros((size, size*size))
        temp_i = np.zeros((size, size*size))

        for j in range(len(w)):

            # dA/dw(j,:) = v(:,j)*(v^-T)(:j)
            partial = np.outer(v[:, j], v_inv_T[:, j]).flatten(order='F')
            # Note that the order of flattening matters, hence argument in flatten()

            # Set jacobian rows
            temp_r[j, :] = np.real(partial)
            temp_i[j, :] = np.imag(partial)

        # Set Jacobian
        derivatives['e_real_long', 'A_long'] = temp_r
        derivatives['e_imag_long', 'A_long'] = temp_i