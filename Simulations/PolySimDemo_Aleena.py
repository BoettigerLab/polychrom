
# import some dependencies 
import pickle, sys, os, h5py, ast, math, copy, time, scipy
import numpy as np
import numpy.matlib
import pandas as pd
# # Python's continual incompetence with finding packages
sys.path.append("C:\Shared\polychrom-shared")
import polychrom
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.simulation import Simulation
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from scipy import spatial  

sys.path.append(r'C:\Users\Alistair\Desktop\code\minrylab-polychrom\Simulations')
# import LEBondUpdater
from LEBondUpdater import bondUpdater 

#---------------------------#
saveFolder = r'Z:\Alistair\PolymerSims\2022-08-16_Demo/'  # Update ME
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)  # only creates folders 1 deep, won't create a full path
    


# ==================== general parameters  =======================


iters = 1 # number of iterations to do



#  ==========Extrusion sim parameters====================
density =  0.2  # density of the PBC box 
N1 = 1000 #  Number of monomers in the polymer
M = 1  # separate chains in the same volume (will interact in trans with sticky sims)
N = N1 * M # number of monomers in the full simulation 
LIFETIME =  50 #  [Imakaev/Mirny use 200 as demo] extruder lifetime
SEPARATION = 90 #80  ave. separation between extruders in monomer units (extruder density) 
ctcfSites =  np.array([0]) #  np.array([0,399,400]) #CTCF site locations  # positioned on HoxA
nCTCF = np.shape(ctcfSites)[0]
ctcfDir = np.zeros(nCTCF) # 0 is bidirectional, 1 is right 2 is left
ctcfCapture = 0.98*np.ones(nCTCF) #  capture probability per block if capture < than this, capture  
ctcfRelease =0.003*np.ones(nCTCF)  # % release probability per block. if capture < than this, release
# compartment labels
oneChainMonomerTypes = np.zeros(N1).astype(int)
oneChainMonomerTypes[200:400] = 1 # mod self interaction
oneChainMonomerTypes[100:150] = 2 # mod self interaction
# create interaction matrix
interactionMatrix = np.array([[0, 0, 0], [0, .2, .1], [0, .1, .4]])  #=== KEY parameter  ===#
# ==== coil globule transition occurs around .7 for monomers of this length

# load prob
loadProb = np.ones([1,N1])  # uniform loading probability
loadProb = numpy.matlib.repmat(loadProb,1,M) # need to replicate and renormalize
loadProb = loadProb/np.sum(loadProb) 
LEFNum =  0 # math.floor(N // SEPARATION )  # make 0 for no LEFs
monomers = N1


# less common parameters
attraction_radius = 1.5  #  try making this larger
num_chains = M  # simulation uses some equivalent chains  (5 in a real sim)
MDstepsPerCohesinStep = 800
smcBondWiggleDist = 0.2
smcBondDist = 0.5
angle_force = 1.5 # most sims ran with 1.5.  0 might have been better

# save pars
saveEveryBlocks = 10   # save every 10 blocks 
restartSimulationEveryBlocks = 100  # blocks per iteration
trajectoryLength =  restartSimulationEveryBlocks #  1000 # time duration of simulation (down from 100,000)


# check that these loaded alright
print(f'LEF count: {LEFNum}')
print('interaction matrix:')
print(interactionMatrix)
print('monomer types:')
print(oneChainMonomerTypes)
print(saveFolder)

for r in range(iters):
    print(r)
    #==================== simulate epigenetic spreading 
    # load polymer
    # files = list_URIs(saveFolder)
    # data = load_URI(files[-1])  # this is the full data structure, it is possible we only want data['pos']

    newFolder = saveFolder+'run'+str(r+1)+'/'
    lefPosFile = newFolder + "LEFPos.h5"
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)
    reporter = HDF5Reporter(folder=newFolder, max_data_length=100)
    print('creating folder')


    #==================================#
    # Run and load 1D simulation
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
    # Initial simulation using fixed input states
    t=0
    LEFsubset = LEFpositions[t*restartSimulationEveryBlocks:(t+1)*restartSimulationEveryBlocks,:,:] # a subset of the total LEF simulation time
    milker = bondUpdater(LEFsubset)
    data = grow_cubic(N,int((N/(density*1.2))**0.333))  # starting conformation
    PBC_width = (N/density)**0.333
    chains = [(N_chain*(k),N_chain*(k+1),True) for k in range(num_chains)]  # subchains as rings
    reporter = HDF5Reporter(folder=newFolder, max_data_length=100)
    a = Simulation(N=N, 
                   error_tol=0.01, 
                   collision_rate=0.02, 
                   integrator ="variableLangevin", 
                   platform="cuda",
                   GPU = "0", 
                   PBCbox=(PBC_width, PBC_width, PBC_width),
                   reporters=[reporter],
                   precision="mixed")  # platform="CPU", 
    a.set_data(data) # initial polymer 
    a.add_force(
        polychrom.forcekits.polymer_chains(
            a,
            chains=chains,
            nonbonded_force_func=polychrom.forces.heteropolymer_SSW,
            nonbonded_force_kwargs={
                'attractionEnergy': 0,  # base attraction energy for all monomers
                'attractionRadius': attraction_radius,
                'interactionMatrix': interactionMatrix,
                'monomerTypes': monomerTypes,
                'extraHardParticlesIdxs': []
            },
            bond_force_kwargs={
                'bondLength': 1,
                'bondWiggleDistance': 0.05
            },
            angle_force_kwargs={
                'k': angle_force
            }
        )
    )
    # ------------ initializing milker; adding bonds ---------
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale
    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0}
    milker.setParams(activeParams, inactiveParams)  
    milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                blocks=restartSimulationEveryBlocks)
    # If your simulation does not start, consider using energy minimization below
    a.local_energy_minimization()  # only do this at the beginning

    # this runs 
    for i in range(restartSimulationEveryBlocks):   # loops over 100     
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

# ---