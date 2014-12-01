import numpy as np
from dedalus2.public import *

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class Incompressible_RT:
    def __init__(self, domain):
        self.domain = domain
        
    def set_problem(self, Reynolds, Prandtl, periodic=False):
        logger.info("Re = {:g}, Pr = {:g}".format(Reynolds, Prandtl))
        
        self.problem = ParsedProblem( axis_names=['x', 'z'],
                                field_names=['u','u_z','w','w_z','T', 'T_z', 'P'],
                                param_names=['Re', 'Pr', 'Pe'])

        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(T) - T_z = 0")

        self.problem.add_equation("dt(w) - 1/Re*(dx(dx(w)) + dz(w_z))  + dz(P) = -(u*dx(w) + w*w_z)")
        self.problem.add_equation("dt(u) - 1/Re*(dx(dx(u)) + dz(u_z))  + dx(P) = -(u*dx(u) + w*u_z)")
        self.problem.add_equation("dt(T) - 1/Pe*(dx(dx(T)) + dz(T_z))          = -(u*dx(T) + w*T_z)")
        self.problem.add_equation("dx(u) + w_z = 0")
            
        self.problem.parameters['Re']  = Reynolds
        self.problem.parameters['Pr']  = Prandtl
        self.problem.parameters['Pe']  = Reynolds*Prandtl # Peclet number

        if not(periodic):
            logger.info("Imposing fixed boundaries, for a single shear layer")
            self.problem.add_left_bc( "T = 0") # unstable: T heavier on top
            self.problem.add_right_bc("T = 1")
            #self.problem.add_left_bc( "T = 1") # stable: T heavier on bottom
            #self.problem.add_right_bc("T = 0")
            #self.problem.add_left_bc( "u = 0.5")  # shear BC
            #self.problem.add_right_bc("u = -0.5")
            self.problem.add_left_bc( "u = 0")     # \vec{u} = 0 BC
            self.problem.add_right_bc("u = 0")
            self.problem.add_left_bc( "w = 0", condition="dx != 0") # for dx=0, w=const, so need to
            self.problem.add_left_bc( "P = 0", condition="dx == 0") # "manually" account for this
            self.problem.add_right_bc("w = 0")                      # by only setting w=0 for dx!=0
        else:
            logger.info("Imposing periodic boundaries")
            logger.info(" -- Applying internal BC to P for k=0")
            self.problem.add_int_bc("P = 0", condition="dx == 0")

        self.problem.expand(self.domain, order=1)

        return self.problem
    
class Boussinesq_RT:
    def __init__(self, domain):
        self.domain = domain
        
    def set_problem(self, Rayleigh_thermal, Prandtl, periodic=False):
        logger.info("Ra_T = {:g}, Pr = {:g}".format(Rayleigh_thermal, Prandtl))
        
        self.problem = ParsedProblem( axis_names=['x', 'z'],
                                field_names=['u','u_z','w','w_z','T', 'T_z', 'P'],
                                param_names=['Ra_T', 'Pr'])

        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(T) - T_z = 0")

        #self.problem.add_equation("1/Pr*dt(w)  - (dx(dx(w)) + dz(w_z)) - Ra_T*T + dz(P) = -1/Pr*(u*dx(w) + w*w_z)")
        self.problem.add_equation("1/Pr*dt(w)  - (dx(dx(w)) + dz(w_z))  + dz(P) = -1/Pr*(u*dx(w) + w*w_z)")
        self.problem.add_equation("1/Pr*dt(u)  - (dx(dx(u)) + dz(u_z))  + dx(P) = -1/Pr*(u*dx(u) + w*u_z)")
        self.problem.add_equation("dt(T) -       (dx(dx(T)) + dz(T_z))          = -u*dx(T) - w*T_z")
        self.problem.add_equation("dx(u) + w_z = 0")

        if not(periodic):
            logger.info("Imposing fixed boundaries")
            self.problem.add_left_bc( "T = 0")
            self.problem.add_right_bc("T = 0")
            self.problem.add_left_bc( "u = 0")
            self.problem.add_right_bc("u = 0")
            self.problem.add_left_bc( "w = 0", condition="dx != 0")
            self.problem.add_left_bc( "P = 0", condition="dx == 0")
            self.problem.add_right_bc("w = 0")
        else:
            logger.info("Imposing periodic boundaries")
            logger.info(" -- Applying internal BC to P for k=0")
            self.problem.add_int_bc("P = 0", condition="dx == 0")
            
        self.problem.parameters['Ra_T']  = Rayleigh_thermal
        self.problem.parameters['Pr']    = Prandtl

        self.problem.expand(self.domain, order=1)

        return self.problem
