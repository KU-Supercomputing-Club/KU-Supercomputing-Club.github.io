---
title: A Quick Introduction to Parallel Programming 
description: A hands-on tutorial to parallel programming.
author: abir
date: 2024-09-17 17:00:00 +0000
categories: [Programming, Tutorial]
tags: [supercomputing, tutorial, programming]
pin: false
math: true
mermaid: true
image:
  path: https://ku-supercomputing-club.github.io/assets/img/commons/ksc_logo.png
  alt: ksc
---

### Introduction
This is a very quick, hands-on lab showcasing parallel programming with Message Passing Interface (MPI).

MPI is a very powerful tool for parallel programming across a network of multi-core systems. MPI can be used in many programming languages, such as C/C++, Python, Octave, etc. This tutorial will use Python via the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) module.

##Setup
Before we get into writing any programs, lets focus on setting up our environment. It is best practice to write Python programs in a virutal environment to avoid installing packaches globally, which is problematic on a shared system like a supercomputing cluster.

Let's first create a folder called `mpi_tutorial` for our environment (you can use any other forder name that you wish). Type in the following into your terminal to create a new `mpi_tutorial` folder:
```bash
mkdir mpi_tutorial
```

Next, let's create the environment at that location:
```bash
python3 -m venv mpi_tutorial/
```

Now, let's enter our environment:
```bash
source mpi_tutorial/bin/activate
```

We will also need to execute our MPI program via mpirun and load the MPI library, which is available in the openmpi module:
```bash
module load openmpi
```

Lastly, we can now start installing dependencies for our Python programs in this environment. We only need mpi4py, so lets install that in our environment:
```bash
pip3 install mpi4py
```

We are done with the initial setup! If you ever exit and enter your session again, you will need to execute the following again (no need to make the mpi_tutorial folder, create an environment, or install mpi4py again):
```bash
source mpi_tutorial/bin/activate
module load openmpi
```

### Basic MPI Usage
Below are a few commonly used MPI commands in action: 
# **`test.py`**
```python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Halo-exchange of rank values via sent/recv
comm.send(rank,dest=(rank+1)%size)
val=comm.recv()
comm.barrier()
print("rank =",rank,"recieved",val)

#Broacast data from last rank to all ranks
arr_size=10
data=np.empty(arr_size,dtype=int)
#Each rank has a local 10-element array filled with their rank value
for i in range(arr_size):
        data[i]=rank
print("Before bcast: rank =",rank,data)
#Now we broadcast from the last rank to all ranks. All ranks should only have arrays filled with size-1
data=comm.bcast(data,root=size-1)
print("After bcast: rank =",rank,data)

#Reduction operation: even ranks store 2/(size/2) at even indices, odd ranks store 1(size/2) at odd indices, then sum all contributions to get [2,1,2,1...
local_data=np.zeros(arr_size,dtype=np.float64)
data=np.zeros(arr_size,dtype=np.float64)
if rank%2==0:
        for i in range(0,arr_size,2):
                local_data[i]=2/(size/2)
else:
        for i in range(1,arr_size,2):
                local_data[i]=1/(size/2)
comm.barrier()
print("Before allreduce: rank =",rank,local_data)
comm.Allreduce([local_data,MPI.DOUBLE],[data,MPI.DOUBLE],op=MPI.SUM)
print("After allreduce: rank =",rank,data)

```

If you want to execute this, type the following into your terminal:
```bash
mpirun --mca btl self,tcp -n 4 python3 test.py
```

### Application
Matrix-matrix multiplication arises in many engineering and computing applications. For this tutorial, we will consider solving $Ax=b$, where we are solving for $b$.

Below is example serial code that computes a matrix-matrix multiplication:
# **`matmul.py`**
```python
import numpy as np
from time import time
import sys


dimX=int(sys.argv[1])
dimY=int(sys.argv[2])
A=np.random.rand(dimX,dimX)
x=np.random.rand(dimX,dimY)
start=time()
b=A@x
end=time()
print("Serial time:",end-start)

```

Below is one way of parallelizing matrix multiplcation across a cluster:
# **`mpi_matmul.py`**
```python
from mpi4py import MPI
import numpy as np
from time import time
import sys

#Get MPI communicator, which allows us to exchange data between cores
comm = MPI.COMM_WORLD

#MPI assigns each core a rank, which is an "identidication" number
rank = comm.Get_rank()

#Get the total number of ranks in our communicator
size = comm.Get_size()

#Number of rows/columns in A
dimX=int(sys.argv[1])

#Number of columns in Y
dimY=int(sys.argv[2])

#Need to initialize local variables across all ranks
A=None
x=None
b=None

#Timing variables
start=None
end=None

#Preprocessing: Only need to generate A and x on a single rank, which saves memory
if rank == 0:
        A=np.random.rand(dimX,dimX)
        x=np.random.rand(dimX,dimY)
        b=np.empty((dimX,dimY),dtype=np.float64) #Allocated enough memory for final result
comm.barrier() #Syncronize

#Start timing
if rank==0:
        start = time()

#Calculate number of rows per rank to process
num_rows_per_rank=np.empty(size,dtype=int)
for i in range(size):
        num_rows_per_rank[i]=dimX//size
remainder=dimX-num_rows_per_rank[0]*size
for i in range(remainder):
        num_rows_per_rank[i]+=1

#Allocate memory for rank-local rows of A
localA=np.empty((num_rows_per_rank[rank],dimX),dtype=np.float64)

#Distribute rows of A to all ranks
comm.Scatterv((A,num_rows_per_rank*dimX),localA,root=0)

#Broadcase X to all ranks
x=comm.bcast(x,root=0)

#Syncronize
comm.barrier()

#Perform rank-local matrix-vector multiplication with rank-local rows of A
localb=localA@x

#Assemble distributed parts of b
comm.Gatherv(localb,(b,num_rows_per_rank*dimY),root=0)

#End time
if rank==0:
        end=time()
        print("Parallel time:",end-start)
```

To execute this MPI program with specific node/core arrangements, you need to write a sbatch script. Below is an example sbatch script that runs a scaling test on up to 16 nodes:
# **`mpi_matmul.sh`**
```bash
#!/bin/bash

#SBATCH --partition=intel # Partition Name (Required)
#SBATCH --ntasks-per-node=1 # Number of cores
#SBATCH --nodes=16 # Number of nodes
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50g # Job memory request
#SBATCH --time=0-08:00:00 # Time limit days-hrs:min:sec
#SBATCH --output=mpi_test\_%j.log # Standard output and error log

dimX=20000
dimY=200

echo 1
python3 matmul.py $dimX $dimY
echo 2
mpirun --mca btl self,tcp -n 2 python3 mpi_matmul.py $dimX $dimY
echo 4
mpirun --mca btl self,tcp -n 4 python3 mpi_matmul.py $dimX $dimY
echo 8
mpirun --mca btl self,tcp -n 8 python3 mpi_matmul.py $dimX $dimY
echo 16
mpirun --mca btl self,tcp -n 16 python3 mpi_matmul.py $dimX $dimY
```

You can execute this sbatch script by typing the following into your terminal:
```bash
sbatch mpi_matmul.sh
```
