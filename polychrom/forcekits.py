import numpy as np
from . import forces

def polymer_chains(
    sim_object,
    chains=[(0, None, False)],

    bond_force_func=forces.harmonic_bonds,
    bond_force_kwargs={'bondWiggleDistance':0.05,
                     'bondLength':1.0},

    angle_force_func=forces.angle_force,
    angle_force_kwargs={'k':0.05},

    nonbonded_force_func=forces.polynomial_repulsive,
    nonbonded_force_kwargs={'trunc':3.0, 
                          'radiusMult':1.},

    except_bonds=True,
):
    """Adds harmonic bonds connecting polymer chains

    Parameters
    ----------
    chains: list of tuples
        The list of chains in format [(start, end, isRing)]. The particle
        range should be semi-open, i.e. a chain (0,3,0) links
        the particles 0, 1 and 2. If bool(isRing) is True than the first
        and the last particles of the chain are linked into a ring.
        The default value links all particles of the system into one chain.

    exceptBonds : bool
        If True then do not calculate non-bonded forces between the
        particles connected by a bond. True by default.
    """
    
    

    force_list = []

    bonds = []
    triplets = []
    newchains = []
    
    for start, end, is_ring in chains:
        end = sim_object.N if (end is None) else end
        newchains.append((start, end, is_ring))
        
        bonds += [(j, j+1) for j in range(start, end - 1)]
        triplets += [(j - 1, j, j + 1) for j in range(start + 1, end - 1)]

        if is_ring:
            bonds.append((start, end-1))
            triplets.append((int(end - 2), int(end - 1), int(start)))
            triplets.append((int(end - 1), int(start), int(start + 1)))
            
    reportDict = {"chains":np.array(newchains, dtype=int), 
                  "bonds": np.array(bonds, dtype=int),
                  "angles": np.array(triplets)
                 }
    for reporter in sim_object.reporters:
        reporter.report("forcekit_polymer_chains", reportDict)
    
    if bond_force_func is not None: 
        force_list.append(
            bond_force_func(sim_object, bonds, **bond_force_kwargs)
        )
    
    if angle_force_func is not None:
        force_list.append(
            angle_force_func(sim_object, triplets, **angle_force_kwargs)
        )
    
    if nonbonded_force_func is not None:
        nb_force = nonbonded_force_func(sim_object, **nonbonded_force_kwargs)
        
        if except_bonds:
            exc = list(set([tuple(i) for i in np.sort(np.array(bonds), axis=1)]))
            if hasattr(nb_force, "addException"):
                print('Exclude neighbouring chain particles from {}'.format(nb_force.name))
                for pair in exc:
                    nb_force.addException(int(pair[0]), int(pair[1]), 0, 0, 0, True)
                    
            # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
            elif hasattr(nb_force, "addExclusion"):
                print('Exclude neighbouring chain particles from {}'.format(nb_force.name))
                for pair in exc:
                    nb_force.addExclusion(int(pair[0]), int(pair[1]))
                    
            print("Number of exceptions:", len(bonds))

        force_list.append(nb_force)

    return force_list

            
