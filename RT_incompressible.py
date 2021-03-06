import numpy as np
import time
import os
import sys
import getopt
import equations

import logging
logger = logging.getLogger(__name__)

from dedalus2.public import *
from dedalus2.tools  import post
from dedalus2.extras import flow_tools
#from dedalus2.extras.checkpointing import Checkpoint

#####################################################################
initial_time = time.time()

logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

# parse command line arguments
Reynolds = 2500  # problem parameters
Prandtl = 1
Lx = 1           # set domain
Lz = 2
tstop = 30       # simulation stop time
tstop_wall = 100 # max walltime limit in hours
nx_tmp = 128     # resolution
nz_tmp = 128
dir_str = ""     # extra string on directory name

# parse command args
try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["Re=", "Pr=", "Lx=", "Lz="
                           "tstop=", "twall-stop=", "nx=", "nz=", "dir-str="])
except getopt.GetoptError:
    print ("\n\n\tBad Command Line Args -- Using Defaults\n\n")
for opt, arg in opts:
    if opt in ("--Re"):
        Reynolds = float(arg)
    elif opt in ("--Pr"):
        Prandtl = float(arg)
    elif opt in ("--Lx"):
        Lx = float(arg)
    elif opt in ("--Lz"):
        Lz = float(arg)
    elif opt in ("--tstop"):
        tstop = float(arg)
    elif opt in ("--twall-stop"):
        tstop_wall = float(arg)
    elif opt in ("--nx"):
        nx_tmp = int(arg)
    elif opt in ("--nz"):
        nz_tmp = int(arg)
    elif opt in ("--dir-str"):
        dir_str = arg

# if dir_str is empty, set it to the Reynolds number
if (dir_str == ""):
    dir_str = str(Reynolds)

nx = np.int(nx_tmp*3/2)
nz = np.int(nz_tmp*3/2)

x_basis = Fourier(nx,   interval=[0., Lx], dealias=2/3)
z_basis = Chebyshev(nz, interval=[0., Lz], dealias=2/3)
domain = Domain([x_basis, z_basis], grid_dtype=np.float64)

# save data in directory named after script
script_name = sys.argv[0].split('.py')[0] # this is path of python executable
script_name = script_name.split("/")[-1]  # this removes any "/" in filename
data_dir_prefix = "/charybdis/toomre/ryor5023/Projects/Rayleigh-Taylor/"
if (dir_str[0] != "_"):
    dir_str = "_" + dir_str
data_dir = data_dir_prefix + script_name + "_" + str(nx_tmp) + "x" + \
           str(nz_tmp) + dir_str + "/"

if domain.distributor.rank == 0:
  if not os.path.exists('{:s}/'.format(data_dir)):
    os.mkdir('{:s}/'.format(data_dir))

RT = equations.Incompressible_RT(domain)
pde = RT.set_problem(Reynolds, Prandtl)

logger.info("Nx = {:g}, Nz = {:g}".format(nx, nz))

ts = timesteppers.RK443
cfl_safety_factor = 0.1*4

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

# initially stable stratification
stable = False
if (stable):
    tanh_width = 0.025
    tanh_center = 0.5
    phi = 0.5*(1-np.tanh((z-tanh_center)/tanh_width))
else:
    tanh_width = 0.025
    tanh_center = 0.5*Lz
    phi = 0.5*(1+np.tanh((z-tanh_center)/tanh_width))

shear = False

A_u = 1

T['g'] = phi
if (shear):
    u['g'] = A_u*(phi-0.5)
else:
    u['g'] = np.zeros((len(z)))
Noise_IC = False
if Noise_IC:
    w['g'] = A_u*1e-1*np.sin(z/Lz)*np.random.randn(*w['g'].shape)
else:
    if (stable):
        w['g'] = -A_u*1e-1*np.sin(z/Lz)*np.sin(2*np.pi*x/Lx)
    else:
        w['g'] = 0.25*A_u*np.cos(2*np.pi*x/Lx)
    
logger.info("Au = {:g}".format(A_u))
logger.info("u = {:g} -- {:g}".format(np.min(u['g']), np.max(u['g'])))
logger.info("T = {:g} -- {:g}".format(np.min(T['g']), np.max(T['g'])))

# integrate parameters

max_dt = 0.1
cfl_cadence = 1
cfl = flow_tools.CFL_conv_2D(solver, max_dt, cfl_cadence=cfl_cadence)

report_cadence = 10
output_time_cadence = 0.05
solver.stop_sim_time = tstop
solver.stop_iteration= np.inf
solver.stop_wall_time = tstop_wall*3600

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


