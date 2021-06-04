# Display options don't work

## py_show doesn't work on windows due to the way file opening is handed
`out = tempfile.NamedTemporaryFile` both creates and opens the file
`open(out.name)` is permitted on Linux but not in Windows for a file already open.

## nglutils fails on import
using Anaconda3 with latest mdtraj 
```
import nglutils.nglutils as ngu
```
Produces error
`from .dcd import DCDTrajectoryFile`
`ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`