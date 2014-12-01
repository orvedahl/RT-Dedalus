import numpy as np
import time
import os
import sys
import equations

import logging
logger = logging.getLogger(__name__)

from dedalus2.public import *
from dedalus2.tools  import post
from dedalus2.extras import flow_tools
#from dedalus2.extras.checkpointing import Checkpoint

initial_time = time.time()

logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

# save data in directory named after script
data_dir = sys.argv[0].split('.py')[0]+'/'

Rayleigh_thermal = -14000.0
Prandtl = 1
# Set domain
Lz = 1
Lx = 3

nx = np.int(64*3/2)
nz = np.int(32*3/2)

x_basis = Fourier(nx,   interval=[0., Lx], dealias=2/3)
z_basis = Chebyshev(nz, interval=[0., Lz], dealias=2/3)
domain = Domain([x_basis, z_basis], grid_dtype=np.float64)

if domain.distributor.rank == 0:
  if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))

#TS = equations.thermosolutal_doubly_diffusive_convection(domain)
KH = equations.Boussinesq_KH(domain)
pde = KH.set_problem(Rayleigh_thermal, Prandtl)

ts = timesteppers.RK443
cfl_safety_factor = 0.2*4

# Build solver
solver = solvers.IVP(pde, domain, ts)

x = domain.grid(0)
z = domain.grid(1)

# initial conditions
u = solver.state['u']
w = solver.state['w']
T = solver.state['T']

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Lz'] = Lz

A0 = 1e-6
# initially stable stratification
tanh_width = 0.025
tanh_center = 0.25
phi = 0.5*(1-np.tanh((z-tanh_center)/tanh_width)*np.tanh((z-0.5-tanh_center)/tanh_width))

A_u = 2000

T['g'] = phi + A0*np.sin(z/Lz)*np.random.randn(*T['g'].shape)
u['g'] = A_u*phi
#w['g'] = A_u*1e-2*np.sin(z/Lz)*np.cos(4*np.pi*x/Lx)
w['g'] = 1e-3*A_u*np.sin(z/Lz)*np.random.randn(*w['g'].shape)

logger.info("A0 = {:g}".format(A0))
logger.info("Au = {:g}".format(A_u))
logger.info("u = {:g} -- {:g}".format(np.min(u['g']), np.max(u['g'])))
logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))

# integrate parameters

max_dt = 1e-1
cfl_cadence = 1
cfl = flow_tools.CFL_conv_2D(solver, max_dt, cfl_cadence=cfl_cadence)

report_cadence = 1
output_time_cadence = 0.1/A_u
solver.stop_sim_time = 1e-2
solver.stop_iteration= np.inf
solver.stop_wall_time = 0.25*3600

logger.info("output cadence = {:g}".format(output_time_cadence))

analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", sim_dt=output_time_cadence, max_writes=20, parallel=False)

analysis_slice.add_task("T", name="T")
analysis_slice.add_task("T - Integrate(T, dx)/Lx", name="T'")
analysis_slice.add_task("u", name="u")
analysis_slice.add_task("w", name="w")
analysis_slice.add_task("(dx(w) - dz(u))**2", name="enstrophy")


do_checkpointing=False
if do_checkpointing:
    checkpoint = Checkpoint(data_dir)
    checkpoint.set_checkpoint(solver, wall_dt=1800)

solver.dt = max_dt/A_u

start_time = time.time()
while solver.ok:

    # advance
    solver.step(solver.dt)
    
    if solver.iteration % cfl_cadence == 0 and solver.iteration>=2*cfl_cadence:
        domain.distributor.comm_world.Barrier()
        solver.dt = cfl.compute_dt(cfl_safety_factor)
    
    # update lists
    if solver.iteration % report_cadence == 0:
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e},'.format(solver.iteration, solver.sim_time, solver.dt)
        logger.info(log_string)
        
end_time = time.time()

# Print statistics
elapsed_time = end_time - start_time
elapsed_sim_time = solver.sim_time
N_iterations = solver.iteration 
logger.info('main loop time: {:e}'.format(elapsed_time))
logger.info('Iterations: {:d}'.format(N_iterations))
logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))

logger.info('beginning join operation')
if do_checkpointing:
    logger.info(data_dir+'/checkpoint/')
    post.merge_analysis(data_dir+'/checkpoint/')
logger.info(analysis_slice.base_path)
post.merge_analysis(analysis_slice.base_path)

if (domain.distributor.rank==0):

    N_TOTAL_CPU = domain.distributor.comm_world.size
    
    # Print statistics
    print('-' * 40)
    total_time = end_time-initial_time
    main_loop_time = end_time - start_time
    startup_time = start_time-initial_time
    print('  startup time:', startup_time)
    print('main loop time:', main_loop_time)
    print('    total time:', total_time)
    print('Iterations:', solver.iteration)
    print('Average timestep:', solver.sim_time / solver.iteration)
    print('scaling:',
          ' {:d} {:d} {:d} {:d} {:d} {:d}'.format(N_TOTAL_CPU, 0, N_TOTAL_CPU,nx, 0, nz),
          ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                            main_loop_time, 
                                                            main_loop_time/solver.iteration, 
                                                            main_loop_time/solver.iteration/(nx*nz), 
                                                            N_TOTAL_CPU*main_loop_time/solver.iteration/(nx*nz)))
    print('-' * 40)


