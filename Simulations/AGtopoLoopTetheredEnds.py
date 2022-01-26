'''
 Polymer simulations
   TetheredEnds
      this version replaces the periodic boundary conditions with a sphereical confinement, and tethers both ends of each 
         polymer to the spherical boundary.
  This version implements simulations of Cohesin-mediated loop extrusion and heterotypic chromatin interactions.
     Loop extruders may not pass one-another (no Z-loops).  
     Loop extruders are blocked by directional CTCF sites, a site may be left-facing, right-facing, or both (2,1,0)
     Loop extruders fall off after a fixed amount time after binding (to do: explore effects of stochasticity here?)
     Loop extruders walk a fixed constant rate (to do: explore effects of stochasticity here?)
     Loop extruders may load uniformly, or with a predefined probability distribution across the simulated polymer 
        this captures the effect of chromatin-specific loading patterns. 
     Heterotypic interactions are governed by an "interaction matrix" that specifies the affinity between each monomer type, 
       itself, and all other monomer types.
       
'''

import sys
import os
import numpy as np
import numpy.matlib
import h5py
import ast
import pandas as pd
import math

from LEBondUpdater import bondUpdater

sys.path.append("C:\Shared\polychrom-shared")
import polychrom
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.simulation import Simulation
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
import time

# ==== Load parameters
parsFile = sys.argv[1]
print(parsFile)
# parsFile = r'T:\2020-11-29_PolychromSims\Test\simPars.txt'

t = pd.read_table(parsFile,delimiter=';',squeeze=True)
parValues = t.values[0]

print('looping through all entries')
for idx in range(len(t.columns)):
    v = t.values[0][idx]
    n = t.columns[idx]
    print(n)
    print(v)

# general parameters 
trajectoryLength = parValues[10] # 10000 # time duration of simulation (down from 100,000)
density = parValues[11] #  0.2  # density of the PBC box
  
   
#  ==========Extrusion sim parameters====================
# there is probably a more elegant way to read in text values than ast.literal_eval, but this works.  
GPU_ID = str(parValues[14]) # this should be a string
repulsionEnergy= parValues[15] # this is just a number
N1 = parValues[0] # Number of monomers in the polymer
M = parValues[1]  # concatinate replicates of polymer end-to-end (fewer TAD borders to assign, can be averaged if desired for LE)
num_chains = M  # simulation uses some equivalent chains  (5 in a real sim)
N = N1 * M # number of monomers in the full simulation 
LIFETIME = parValues[2]  # 200 [Imakaev/Mirny use 200 as demo] extruder lifetime
SEPARATION = parValues[3] # 500 ave. separation between extruders in monomer units (extruder density) 
ctcfSites = np.array(ast.literal_eval(parValues[4])) # CTCF site locations [80,280,420,550,700,850] # positioned on HoxA
ctcfDir = np.array(ast.literal_eval(parValues[13]))
ctcfCapture = np.array(ast.literal_eval(parValues[5])) # 0.9 80% capture probability per block if capture < than this, capture  
ctcfRelease = np.array(ast.literal_eval(parValues[6])) # 0.003 % release probability per block. if capture < than this, release
interactionMatrix = np.array(ast.literal_eval(parValues[7]))
saveFolder = parValues[8]
oneChainMonomerTypes =  np.array(ast.literal_eval(parValues[9])) # compartment labels
if len(oneChainMonomerTypes) != N1:
    oneChainMonomerTypes = np.zeros(N1).astype(int)


# need to replicate and renormalize
loadProb = np.array(ast.literal_eval(parValues[12] ))# discrete probability distribution that cohesin loads at site N
loadProb = numpy.matlib.repmat(loadProb,1,M)
loadProb = loadProb/np.sum(loadProb) 

if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

lefPosFile = saveFolder + "LEFPos.h5"
LEFNum = max(0,math.floor(N // SEPARATION )-1)

# tethering parameters
#    [should probably rewrite this]
#   select M random points on the edge of sphere
r = 1.05*(3 * N/ (4 * np.pi * density)) ** (1 / 3.0)
theta =np.random.rand(num_chains,1)*np.pi
psi = np.random.rand(num_chains,1)*2*np.pi
x= r*np.cos(psi)*np.sin(theta)
y= r*np.sin(psi)*np.sin(theta)
z= r*np.cos(theta)
end_tethers = np.concatenate((x,y,z),axis=1).tolist() 


# less common parameters
attraction_radius = 1.5
MDstepsPerCohesinStep = 800
smcBondWiggleDist = 0.2
smcBondDist = 0.5
saveEveryBlocks = 100   # save every 10 blocks (saving every block is now too much almost)
restartSimulationEveryBlocks = 100

# check that these loaded alright
print(f'LEF count: {LEFNum}')
print('interaction matrix:')
print(interactionMatrix)
print('monomer types:')
print(oneChainMonomerTypes)
print(saveFolder)
print('Starting simulation')


#==================================#
# Run 
#=================================#
import polychrom.lib.extrusion1Dv2 as ex1D # 1D classes 
ctcfLeftRelease = {}
ctcfRightRelease = {}
ctcfLeftCapture = {}
ctcfRightCapture = {}

# should modify this to allow directionality
for i in range(M): # loop over chains (this variable needs a better name Max)
    for t in range(len(ctcfSites)):
        pos = i * N1 + ctcfSites[t] 
        if ctcfDir[t] == 0:
            ctcfLeftCapture[pos] = ctcfCapture[t]  # if random [0,1] is less than this, capture
            ctcfLeftRelease[pos] = ctcfRelease[t]  # if random [0,1] is less than this, release
            ctcfRightCapture[pos] = ctcfCapture[t]
            ctcfRightRelease[pos] = ctcfRelease[t]
        elif ctcfDir[t] == 1: # stop Cohesin moving toward the right  
            ctcfLeftCapture[pos] = 0  
            ctcfLeftRelease[pos] = 1  
            ctcfRightCapture[pos] = ctcfCapture[t]
            ctcfRightRelease[pos] = ctcfRelease[t]
        elif ctcfDir[t] == 2:
            ctcfLeftCapture[pos] = ctcfCapture[t]  # if random [0,1] is less than this, capture
            ctcfLeftRelease[pos] = ctcfRelease[t]  # if random [0,1] is less than this, release
            ctcfRightCapture[pos] = 0
            ctcfRightRelease[pos] = 1
       
args = {}
args["ctcfRelease"] = {-1:ctcfLeftRelease, 1:ctcfRightRelease}
args["ctcfCapture"] = {-1:ctcfLeftCapture, 1:ctcfRightCapture}        
args["N"] = N 
args["LIFETIME"] = LIFETIME
args["LIFETIME_STALLED"] = LIFETIME  # no change in lifetime when stalled 

occupied = np.zeros(N)
occupied[0] = 1  # (I think this is just prevent the cohesin loading at the end by making it already occupied)
occupied[-1] = 1 # [-1] is "python" for end
cohesins = []

print('starting simulation with N LEFs=')
print(LEFNum)
for i in range(LEFNum):
    ex1D.loadOneFromDist(cohesins,occupied, args,loadProb) # load the cohesins 


with h5py.File(lefPosFile, mode='w') as myfile:
    
    dset = myfile.create_dataset("positions", 
                                 shape=(trajectoryLength, LEFNum, 2), 
                                 dtype=np.int32, 
                                 compression="gzip")
    steps = 100    # saving in 50 chunks because the whole trajectory may be large 
    bins = np.linspace(0, trajectoryLength, steps, dtype=int) # chunks boundaries 
    for st,end in zip(bins[:-1], bins[1:]):
        cur = []
        for i in range(st, end):
            ex1D.translocate(cohesins, occupied, args,loadProb)  # actual step of LEF dynamics 
            positions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
            cur.append(positions)  # appending current positions to an array 
        cur = np.array(cur)  # when we finished a block of positions, save it to HDF5 
        dset[st:end] = cur
    myfile.attrs["N"] = N
    myfile.attrs["LEFNum"] = LEFNum
    
#=========== Load LEF simulation ===========#
trajectory_file = h5py.File(lefPosFile, mode='r')
LEFNum = trajectory_file.attrs["LEFNum"]  # number of LEFs
LEFpositions = trajectory_file["positions"]  # array of LEF positions  
steps = MDstepsPerCohesinStep # MD steps per step of cohesin  (set to ~800 in real sims)
Nframes = LEFpositions.shape[0] # length of the saved trajectory (>25000 in real sims)
print(f'Length of the saved trajectory: {Nframes}')
block = 0  # starting block 

# test some properties 
# assertions for easy managing code below 
assert (Nframes % restartSimulationEveryBlocks) == 0 
assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
simInitsTotal  = (Nframes) // restartSimulationEveryBlocks
# concatinate monomers if needed
if len(oneChainMonomerTypes) != N:
    monomerTypes = np.tile(oneChainMonomerTypes, num_chains)
else:
    monomerTypes = oneChainMonomerTypes
    
N_chain = len(oneChainMonomerTypes)  
N = len(monomerTypes)
print(f'N_chain: {N_chain}')  # ~8000 in a real sim
print(f'N: {N}')   # ~40000 in a real sim
N_traj = trajectory_file.attrs["N"]
print(f'N_traj: {N_traj}')
assert N == trajectory_file.attrs["N"]
print(f'Nframes: {Nframes}')
print(f'simInitsTotal: {simInitsTotal}')
#==============================================================#
#                  RUN 3D simulation                              #
#==============================================================#
milker = bondUpdater(LEFpositions)
data = grow_cubic(N,int((N/(density*1.2))**0.333))  # starting conformation
reporter = HDF5Reporter(folder=saveFolder, max_data_length=50)
# PBC_width = (N/density)**0.333
chains = [(N_chain*(k),N_chain*(k+1),0) for k in range(num_chains)]

for iteration in range(simInitsTotal):
    a = Simulation(N=N, 
                   error_tol=0.01, 
                   collision_rate=0.01, 
                   integrator ="variableLangevin", 
                   platform="cuda",
                   GPU = GPU_ID, 
                   PBCbox=False, # (PBC_width, PBC_width, PBC_width),
                   reporters=[reporter],
                   precision="mixed")  # platform="CPU", 
    a.set_data(data)
    a.add_force(
        polychrom.forcekits.polymer_chains(
            a,
            chains=chains,
            nonbonded_force_func=polychrom.forces.grosberg_repulsive_force,
            bond_force_kwargs={
                'bondLength': 1,
                'bondWiggleDistance': 0.05
            },
            angle_force_kwargs={
                'k': 0.5  # has been 1.5, methinks should be 0
            }
        )
    )
    a.add_force(polychrom.forces.spherical_confinement(a,density=density))
    a.add_force(polychrom.forces.tether_particles(a,[0,N-1],positions=end_tethers,k=30))  # tether ends of polymer)
    
    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale

    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0}
    milker.setParams(activeParams, inactiveParams)
     
    # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                blocks=restartSimulationEveryBlocks)

    # If your simulation does not start, consider using energy minimization below
    if iteration == 0:
        a.local_energy_minimization() 
    else:
        a._apply_forces()
    
    for i in range(restartSimulationEveryBlocks):        
        if i % saveEveryBlocks == (saveEveryBlocks - 1):  
            a.do_block(steps=steps)
        else:
            a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)
        if i < restartSimulationEveryBlocks - 1: 
            curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
    data = a.get_data()  # save data and step, and delete the simulation
    del a
    
    reporter.blocks_only = True  # Write output hdf5-files only for blocks
    
    time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

reporter.dump_data()