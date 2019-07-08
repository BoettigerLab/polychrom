import re
import itertools
from collections.abc import Iterable

import numpy as np 

import simtk.openmm as openmm
import simtk.unit as units

nm = units.meter * 1e-9
fs = units.second * 1e-15
ps = units.second * 1e-12


def _prepend_force_name_to_params(force):
    if not hasattr(force, 'getEnergyFunction'):
        return

    energy = force.getEnergyFunction()
    if hasattr(force, 'getNumGlobalParameters'):
        for i in range(force.getNumGlobalParameters()):
            old_name = force.getGlobalParameterName(i)
            new_name = force.name + '_' + old_name
            force.setGlobalParameterName(i, new_name)
            energy = re.sub(f'(?<!\w){old_name}(?!\w)', new_name, energy)

    force.setEnergyFunction(energy)


def _to_array_1d(scalar_or_array, arrlen, dtype=float):
    if not hasattr(scalar_or_array, "__iter__"):
        outarr = np.full(arrlen, scalar_or_array, dtype)
    else:
        outarr = np.asarray(scalar_or_array, dtype=dtype)
    
    if len(outarr) != arrlen:
        raise ValueError('The length of the array differs from the expected one!')
        
    return outarr


def harmonicBonds(sim_object,
                  bonds,
                  bondWiggleDistance=0.05,
                  bondLength=1.0,
                  name="HarmonicBonds",
                  ):
    """Adds harmonic bonds

    Parameters
    ----------
    
    bonds : iterable of (int, int)
        Pairs of particle indices to be connected with a bond.
    bondWiggleDistance : float or iterable of float
        Average displacement from the equilibrium bond distance.
        Can be provided per-particle.
    bondLength : float or iterable of float
        The length of the bond.
        Can be provided per-particle.
    """
    
    force =  openmm.HarmonicBondForce()
    force.name = name

    bondLength = _to_array_1d(bondLength, len(bonds)) * sim_object.length_scale
    bondWiggleDistance = _to_array_1d(bondWiggleDistance, len(bonds)) * sim_object.length_scale
    
    # using kbondScalingFactor because force accepts parameters with units
    kbond = sim_object.kbondScalingFactor / (bondWiggleDistance ** 2)  

    for bond_idx, (i, j) in enumerate(bonds):
        if (i >= sim_object.N) or (j >= sim_object.N):
            raise ValueError("\nCannot add bond with monomers %d,%d that"\
            "are beyound the polymer length %d" % (i, j, sim_object.N))
        
        force.addBond(int(i), 
                      int(j), 
                      float(bondLength[bond_idx]), 
                      float(kbond[bond_idx]))
        
    return force
    
    
def FENEBonds(sim_object,
              bonds,
              bondWiggleDistance=0.05,
              bondLength=1.0,
              name="FENEBonds",
              ):
    """Adds harmonic bonds

    Parameters
    ----------
    
    bonds : iterable of (int, int)
        Pairs of particle indices to be connected with a bond.
    bondWiggleDistance : float
        Average displacement from the equilibrium bond distance.
        Can be provided per-particle.
    bondLength : float
        The length of the bond.
        Can be provided per-particle.
    """
    
    energy = f"(1. / wiggle) * univK * "\
                f"(sqrt((r-r0 * conlen)* "\
                f" (r - r0 * conlen) + a * a) - a)"
    force = openmm.CustomBondForce(energy)
    force.name = name
    
    force.addPerBondParameter("wiggle")
    force.addPerBondParameter("r0")
    force.addGlobalParameter("univK", sim_object.kT / sim_object.conlen)
    force.addGlobalParameter("a", 0.02 * sim_object.conlen)
    force.addGlobalParameter("conlen", sim_object.conlen)

    bondLength = _to_array_1d(bondLength, len(bonds)) * sim_object.length_scale
    bondWiggleDistance = _to_array_1d(bondWiggleDistance, len(bonds)) * sim_object.length_scale
    
    for bond_idx, (i, j) in enumerate(bonds):
        if (i >= sim_object.N) or (j >= sim_object.N):
            raise ValueError("\nCannot add bond with monomers %d,%d that"\
            "are beyound the polymer length %d" % (i, j, sim_object.N))
        
        force.addBond(int(i), 
                             int(j), 
                             [float(bondWiggleDistance[bond_idx]), 
                              float(bondLength[bond_idx])]) 
        
    return force


def angleForce(
        sim_object, 
        triplets,
        k=1.5,
        name='Angle'):
    """Adds harmonic angle bonds. k specifies energy in kT at one radian
    If k is an array, it has to be of the length N.
    Xth value then specifies stiffness of the angle centered at
    monomer number X.
    Values for ends of the chain will be simply ignored.

    Parameters
    ----------

    k : float or list of length N
        Stiffness of the bond.
        If list, then determines the stiffness of the i-th triplet
        Potential is k * alpha^2 * 0.5 * kT
    """
    
    k = _to_array_1d(k, len(triplets)) 
        
    energy = "kT*angK * (theta - 3.141592) * (theta - 3.141592) * (0.5)"
    force = openmm.CustomAngleForce(energy)
    force.name = name
    
    force.addGlobalParameter("kT", sim_object.kT)
    force.addPerAngleParameter("angK")

    for triplet_idx, (p1, p2, p3) in enumerate(triplets):
        force.addAngle(p1, p2, p3, [k[triplet_idx]])
    
    return force


def polynomialRepulsiveForce(
        sim_object, 
        trunc=3.0, 
        radiusMult=1.,
        name='PolynomialRepulsive'
    ):
    """This is a simple polynomial repulsive potential. It has the value
    of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
    together with its first derivative at r=1.0.

    Parameters
    ----------

    trunc : float
        the energy value around r=0

    """
    radius = sim_object.conlen * radiusMult
    nbCutOffDist = radius
    repul_energy = (
        "rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;")

    force = openmm.CustomNonbondedForce(repul_energy)
    force.name = name

    force.addGlobalParameter('REPe', trunc * sim_object.kT)
    force.addGlobalParameter('REPsigma', radius)
    # Coefficients for x^8*(x*x-1)
    # force.addGlobalParameter('emin12', 256.0 / 3125.0)
    # force.addGlobalParameter('rmin12', 2.0 / np.sqrt(5.0))
    # Coefficients for x^12*(x*x-1)
    force.addGlobalParameter('emin12', 46656.0 / 823543.0)
    force.addGlobalParameter('rmin12', np.sqrt(6.0 / 7.0))

    for _ in range(sim_object.N):
        force.addParticle(())

    force.setCutoffDistance(nbCutOffDist)
    
    return force


def smoothSquareWellForce(sim_object,
    repulsionEnergy=3.0, repulsionRadius=1.,
    attractionEnergy=0.5, attractionRadius=2.0,
    name='SmoothSquareWell'
    ):
    """
    This is a simple and fast polynomial force that looks like a smoothed
    version of the square-well potential. The energy equals `repulsionEnergy`
    around r=0, stays flat until 0.6-0.7, then drops to zero together
    with its first derivative at r=1.0. After that it drop down to
    `attractionEnergy` and gets back to zero at r=`attractionRadius`.

    The energy function is based on polynomials of 12th power. Both the
    function and its first derivative is continuous everywhere within its
    domain and they both get to zero at the boundary.

    Parameters
    ----------

    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionEnergy: float
        the maximal range of the attractive part of the potential.

    """
    nbCutOffDist = sim_object.conlen * attractionRadius
    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep = rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        ""
        "Eattr = - rshft12 * (rshft2 - 1.0) * ATTRe / emin12 - ATTRe;"
        "rshft12 = rshft4 * rshft4 * rshft4;"
        "rshft4 = rshft2 * rshft2;"
        "rshft2 = rshft * rshft;"
        "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12"

        )
    
    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.addGlobalParameter('REPe', repulsionEnergy * sim_object.kT)
    force.addGlobalParameter('REPsigma', repulsionRadius * sim_object.conlen)

    force.addGlobalParameter('ATTRe', attractionEnergy * sim_object.kT)
    force.addGlobalParameter('ATTRdelta',
        sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0)
    # Coefficients for the minimum of x^12*(x*x-1)
    force.addGlobalParameter('emin12', 46656.0 / 823543.0)
    force.addGlobalParameter('rmin12', np.sqrt(6.0 / 7.0))

    for _ in range(sim_object.N):
        force.addParticle(())

    force.setCutoffDistance(nbCutOffDist)
    
    return force

    
def selectiveSSWForce(sim_object,
    stickyParticlesIdxs,
    extraHardParticlesIdxs,
    repulsionEnergy=3.0,
    repulsionRadius=1.,
    attractionEnergy=3.0,
    attractionRadius=1.5,
    selectiveRepulsionEnergy=20.0,
    selectiveAttractionEnergy=1.0,
    name='SelectiveSSW'):
    """
    This is a simple and fast polynomial force that looks like a smoothed
    version of the square-well potential. The energy equals `repulsionEnergy`
    around r=0, stays flat until 0.6-0.7, then drops to zero together
    with its first derivative at r=1.0. After that it drop down to
    `attractionEnergy` and gets back to zero at r=`attractionRadius`.

    The energy function is based on polynomials of 12th power. Both the
    function and its first derivative is continuous everywhere within its
    domain and they both get to zero at the boundary.

    This is a tunable version of SSW:
    a) You can specify the set of "sticky" particles. The sticky particles
    are attracted only to other sticky particles.
    b) You can select a subset of particles and make them "extra hard".

    This force was used two-ways. First was to make a small subset of particles very sticky. 
    In that case, it is advantageous to make the sticky particles and their neighbours
    "extra hard" and thus prevent the system from collapsing.

    Another useage is to induce phase separation by making all B monomers sticky. In that case, 
    extraHard particles may not be needed at all, because the system would not collapse on itsim_object. 


    Parameters
    ----------

    stickyParticlesIdxs: list of int
        the list of indices of the "sticky" particles. The sticky particles
        are attracted to each other with extra `selectiveAttractionEnergy`
    extraHardParticlesIdxs : list of int
        the list of indices of the "extra hard" particles. The extra hard
        particles repel all other particles with extra
        `selectiveRepulsionEnergy`
    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the maximal range of the attractive part of the potential.
    selectiveRepulsionEnergy: float
        the EXTRA repulsion energy applied to the "extra hard" particles
    selectiveAttractionEnergy: float
        the EXTRA attraction energy applied to the "sticky" particles
    """

    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep = rsc12 * (rsc2 - 1.0) * REPeTot / emin12 + REPeTot;"  # + ESlide;"
        "REPeTot = REPe + (ExtraHard1 + ExtraHard2) * REPeAdd;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        ""
        "Eattr = - rshft12 * (rshft2 - 1.0) * ATTReTot / emin12 - ATTReTot;"
        "ATTReTot = ATTRe + min(Sticky1, Sticky2) * ATTReAdd;"
        "rshft12 = rshft4 * rshft4 * rshft4;"
        "rshft4 = rshft2 * rshft2;"
        "rshft2 = rshft * rshft;"
        "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12;"
        ""
        )

    if selectiveRepulsionEnergy == float('inf'):
        energy += (
        "REPeAdd = 4 * ((REPsigma / (2.0^(1.0/6.0)) / r)^12 - (REPsigma / (2.0^(1.0/6.0)) / r)^6) + 1;"
        )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.setCutoffDistance(attractionRadius * sim_object.conlen)

    force.addGlobalParameter('REPe', repulsionEnergy * sim_object.kT)
    if selectiveRepulsionEnergy != float('inf'):
        force.addGlobalParameter('REPeAdd', selectiveRepulsionEnergy * sim_object.kT)
    force.addGlobalParameter('REPsigma', repulsionRadius * sim_object.conlen)

    force.addGlobalParameter('ATTRe', attractionEnergy * sim_object.kT)
    force.addGlobalParameter('ATTReAdd', selectiveAttractionEnergy * sim_object.kT)
    force.addGlobalParameter('ATTRdelta',
        sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0)

    # Coefficients for x^12*(x*x-1)
    force.addGlobalParameter('emin12', 46656.0 / 823543.0)
    force.addGlobalParameter('rmin12', np.sqrt(6.0 / 7.0))

    force.addPerParticleParameter("Sticky")
    force.addPerParticleParameter("ExtraHard")

    counts = np.bincount(stickyParticlesIdxs, minlength=sim_object.N)

    for i in range(sim_object.N):
        force.addParticle(
            (float(counts[i]),
             float(i in extraHardParticlesIdxs)))

    return force


def cylindricalConfinement(
    sim_object, 
    r, 
    bottom=None, 
    k=0.1, 
    top=9999,
    name="CylindricalConfinement"):
    """As it says."""

    if bottom == True:
        warnings.warn(DeprecationWarning(
            "Use bottom=0 instead of bottom = True! "))
        bottom = 0

    if bottom is not None:
        force = openmm.CustomExternalForce(
            "step(r-aa) * kb * (sqrt((r-aa)*(r-aa) + t*t) - t)"
            "+ step(-z + bot) * kb * (sqrt((z - bot)^2 + t^2) - t) "
            "+ step(z - top) * kb * (sqrt((z - top)^2 + t^2) - t);"
            "r = sqrt(x^2 + y^2 + tt^2)")
    else:
        force = openmm.CustomExternalForce(
            "step(r-aa) * kb * (sqrt((r-aa)*(r-aa) + t*t) - t);"
            "r = sqrt(x^2 + y^2 + tt^2)")
    force.name = name

    for i in range(sim_object.N):
        force.addParticle(i, [])
    force.addGlobalParameter("kb", k * sim_object.kT / nm)
    force.addGlobalParameter("top", top * sim_object.conlen)
    if bottom is not None:
        force.addGlobalParameter("bot", bottom * sim_object.conlen)
    force.addGlobalParameter("kt", sim_object.kT)
    force.addGlobalParameter("aa", (r - 1. / k) * nm)
    force.addGlobalParameter("t", (1. / (10 * k)) * nm)
    force.addGlobalParameter("tt", 0.01 * nm)
    
    return force

    
def sphericalConfinement(sim_object,
            r="density",  # radius... by default uses certain density
            k=5.,  # How steep the walls are
            density=.3,    # target density, measured in particles
                           # per cubic nanometer (bond size is 1 nm)
            name='SphericalConfinement'
            ):
    """Constrain particles to be within a sphere.
    With no parameters creates sphere with density .3

    Parameters
    ----------
    r : float or "density", optional
        Radius of confining sphere. If "density" requires density,
        or assumes density = .3
    k : float, optional
        Steepness of the confining potential, in kT/nm
    density : float, optional, <1
        Density for autodetection of confining radius.
        Density is calculated in particles per nm^3,
        i.e. at density 1 each sphere has a 1x1x1 cube.
    """

    force = openmm.CustomExternalForce(
        "step(r-aa) * kb * (sqrt((r-aa)*(r-aa) + t*t) - t); "
        "r = sqrt(x^2 + y^2 + z^2 + tt^2)")
    force.name = name

    for i in range(sim_object.N):
        force.addParticle(i, [])
    if r == "density":
        r = (3 * sim_object.N / (4 * 3.141592 * density)) ** (1 / 3.)

    if sim_object.verbose == True:
        print("Spherical confinement with radius = %lf" % r)
    # assigning parameters of the force
    force.addGlobalParameter("kb", k * sim_object.kT / nm)
    force.addGlobalParameter("aa", (r - 1. / k) * nm)
    force.addGlobalParameter("t", (1. / k) * nm / 10.)
    force.addGlobalParameter("tt", 0.01 * nm)

    ## TODO: move 'r' elsewhere?..
    sim_object.sphericalConfinementRadius = r

    return force


def tetherParticles(
        sim_object, 
        particles, 
        k=30, 
        positions="current",
        name="Tethers"
        ):
    """tethers particles in the 'particles' array.
    Increase k to tether them stronger, but watch the system!

    Parameters
    ----------

    particles : list of ints
        List of particles to be tethered (fixed in space).
        Negative values are allowed.
    k : int, optional
        The steepness of the tethering potential.
        Values >30 will require decreasing potential, but will make tethering 
        rock solid.
        Can be provided as a vector [kx, ky, kz].
    """
    
    energy = "kx * (x - x0)^2 + ky * (y - y0)^2 + kz * (z - z0)^2"
    force = openmm.CustomExternalForce(energy)
    force.name = name

    # assigning parameters of the force

    if isinstance(k, Iterable):
        k = list(k)
        if len(k) != 3:
            raise ValueError('k must either be a scalar or a 3D vector!')
        kx, ky, kz = k
    else:
        kx, ky, kz = k, k, k

    force.addGlobalParameter("kx", kx * sim_object.kT / nm)
    force.addGlobalParameter("ky", ky * sim_object.kT / nm)
    force.addGlobalParameter("kz", kz * sim_object.kT / nm)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    _prepend_force_name_to_params(force)

    particles = [sim_object.N+i if i<0 else i for i in particles]

    if positions == "current":
        positions = [sim_object.data[i] for i in particles]
    else:
        positions = sim_object.addUnits(positions)

    for i, pos in zip(particles, positions):  # adding all the particles on which force acts
        i = int(i)
        force.addParticle(i, list(pos))
        if sim_object.verbose == True:
            print("particle %d tethered! " % i)
    
    return force
            
    
def pullForce(
        sim_object, 
        particles, 
        force_vecs,
        name="Pull"):
    """
    adds force pulling on each particle
    particles: list of particle indices
    force_vecs: list of forces [[f0x,f0y,f0z],[f1x,f1y,f1z], ...]
    if there are fewer forces than particles forces are padded with forces[-1]
    """
    force = openmm.CustomExternalForce(
        "x * fx + y * fy + z * fz")
    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")

    for num, force_vec in itertools.zip_longest(particles, force_vecs, fillvalue=forces[-1]):
        force_vec = [float(f) * (sim_object.kT / sim_object.conlen) for f in force_vec]
        force.addParticle(num, force_vec)
    
    return force


