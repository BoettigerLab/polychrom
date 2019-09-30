import numpy as np 
import warnings
import h5py 
import glob
import simtk.openmm 
import os 
import shutil


def _read_h5_group(gr):
    """
    Reads all attributes of an HDF5 group, and returns a dict of them
    """
    result = {i:j[:] for i,j in gr.items()}
    for i,j in gr.attrs.items():
        result[i] = j
    return result     


def _convert_to_hdf5_array(data):
    """
    Attempts to convert data to HDF5 compatible array
    or to an HDF5 attribute compatible entity (str, number)
    
    Does its best at determining if this is a "normal" 
    object (str, int, float), or an array. 
    
    Right now, if something got converted to a numpy object, 
    it is discarded and not saved in any way. 
    We could think about pickling those cases, or JSONing them... 
    """
    if type(data) == str:
        data = np.array(data, dtype="S")
    data = np.array(data)    
    
    if data.dtype ==  np.dtype("O"):        
        return None,None

    if len(data.shape) == 0:
        return ("item", data.item())
    else:
        return ("ndarray", data)

    
    
def _write_group(dataDict, group, dset_opts={"compression":"gzip"}):
    """
    Writes a dictionary of elements to an HDF5 group
    Puts all "items" into attrs, and all ndarrays into datasets 
    
    dset_opts is a dictionary of arguments passed to create_dataset function
    (compression would be here for example)
    """
    for name,data in dataDict.items():
        datatype, converted = _convert_to_hdf5_array(data)
        if datatype is None:
            warnings.warn(f"Could not convert record {name}")            
        elif datatype == "item":
            group.attrs[name] = data
        elif datatype == "ndarray":
            group.create_dataset(name, data=data, **dset_opts)
        else:
            raise ValueError("Unknown datatype")


        

def list_URIs(folder, readError=True):    
    """
    Takes a trajectory folder and makes a list of dset paths for each block. 
    It is needed because now we store multiple blocks per file. 
    
    It should be compatible with polymerutils.load
    """
    
    files = glob.glob(os.path.join(folder, "blocks_*-*.h5"))
    if len(files) == 0:
        raise ValueError(f"No files found in folder {folder}")
    filenames = {}
    for file in files:
        try:
            f1 = h5py.File(file,'r')
        except:
            if readError:
                raise ValueError(f"Cannot read file {file}")
        sted = os.path.split(file)[-1].split("_")[1].split(".h5")[0]
        st,end = [int(i) for i in sted.split("-")]
        for i in range(st,end+1):
            if i in filenames:
                raise ValueError(f"Block {i} exists more than once")
            filenames[i] =  file+f"::{i}"
    return [i[1] for i in sorted(filenames.items(), key=lambda x:int(x[0]))]

def load_URI(dset_path):
    """
    Loads a single block of the simulation using address provided by list_filenames
    dset_path should be 
    
    /path/to/trajectory/folder/blocks_X-Y.h5::Z    
    
    where Z is the block number 
    """
    
    fname, group = dset_path.split("::")
    with h5py.File(fname, mode='r') as myfile:        
        return _read_h5_group(myfile[group])

def save_hdf5_file(filename, data_dict, dset_opts={"compression":"gzip"}):
    """
    Saves data_dict to filename 
    """
    with h5py.File(filename) as file: 
        _write_group(data_dict, file, dset_opts=dset_opts)

    
def load_hdf5_file(fname):
    """
    Loads a saved HDF5 files, reading all datasets and attributes. 
    We save arrays as datasets, and regular types as attributes in HDF5
    """
    with h5py.File(fname, mode='r') as myfile:        
        return _read_h5_group(myfile)
    
            
            
class HDF5Reporter(object):
    def __init__(self, folder, max_data_length=50, 
                 h5py_dset_opts={"compression":"gzip"}, 
                 overwrite=False, 
                 blocks_only=False
                ):
        """
        
        """
        prefixes = ["blocks", "applied_forces", "initArgs", "starting_conformation", "energy_minimization"]
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        if overwrite: 
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                if os.path.isfile(file_path):                    
                    for prefix in prefixes:
                        if the_file.startswith(prefix):
                            os.remove(file_path)
                else:
                    raise IOError("Subfolder in traj folder; not deleting. Ensure folder is correct and delete manually. ")
                        
                    
        if len(os.listdir(folder)) != 0:
            for the_file in os.listdir(folder):            
                for prefix in prefixes:
                    if the_file.startswith(prefix):
                        raise RuntimeError(f"folder {folder} is not empty")
        self.counter = {}
        self.datas = {}
        self.max_data_length = max_data_length
        self.h5py_dset_opts = h5py_dset_opts
        self.folder = folder
        self.blocks_only = blocks_only
        

    def report(self, name, values):
        count = self.counter.get(name, 0)
        
        
        if name not in ["data"]:
            if not self.blocks_only:
                filename = f"{name}_{count}.h5"
                with h5py.File(os.path.join(self.folder,filename)) as file: 
                    _write_group(values, file, dset_opts=self.h5py_dset_opts)
                
        else:
            self.datas[count] = values 
            if len(self.datas) == self.max_data_length:
                self.dump_data()
        self.counter[name] = count + 1
                
                
    def dump_data(self):
        if len(self.datas) > 0:
            cmin = min(self.datas.keys())
            cmax = max(self.datas.keys())
            filename = f"blocks_{cmin}-{cmax}.h5"
            with h5py.File(os.path.join(self.folder,filename)) as file: 
                for count, values in self.datas.items():                     
                    gr = file.create_group(str(count))
                    _write_group(values, gr, dset_opts=self.h5py_dset_opts)
            self.datas = {}

            

            
            
                