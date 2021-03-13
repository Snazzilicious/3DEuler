
import numpy as np
from scipy.sparse.linalg import spsolve
import sys
import importlib

if 'MeshStructs' in sys.modules:
	import MeshStructs as MS
	importlib.reload(MS)
else:
	import MeshStructs as MS

if 'FluxFunctions' in sys.modules:
	import FluxFunctions as FF
	importlib.reload(FF)
else:
	import FluxFunctions as FF


""" Set Control Variables """
visc = 1.0
numIts = 10
numSubIts = 10
NUM_VARS = 5

# Pick a mesh file
filename = "Mesh/2DCircle.su2"

# Load in the Mesh
print("Loading the Mesh", flush=True)
MS.parseMesh(filename)


""" Set free stream conditions """
M_inf = 1.25
AoA = 0*np.pi/180.0

# don't change these
Vinf = np.array([np.cos(AoA), np.sin(AoA)])
en_inf = ( 1.0 / ( FF.gamma*(FF.gamma-1)*M_inf*M_inf ) )



# Shortucts to Variables
rhoIndices = np.arange(MS.nNodes)
v1Indices=np.arange(MS.nNodes) + MS.nNodes
v2Indices=np.arange(MS.nNodes) + 2*MS.nNodes
v3Indices=np.arange(MS.nNodes) + 3*MS.nNodes
enIndices=np.arange(MS.nNodes) + 4*MS.nNodes

outerIndices = []
for name in ['Inflow', 'Outflow', 'Top', 'Bottom','Left','Right']:
	gInd = MS.groupNames.index(name)
	outerIndices = list( set(outerIndices) | set(MS.groupMembers[gInd]) ) # logical union
outerIndices.sort()
outerIndices = np.array(outerIndices)

bodyInd = MS.groupNames.index('Body')
bodyIndices = np.array(MS.groupMembers[bodyInd]) # Don't sort these

nBodyNodes = len(bodyIndices)

# Selecting which direction to enforce no penetration in
whichDir = np.argmax(np.abs(MS.bodyNodeNormals),axis=1)

bodyBCInds = np.zeros(nBodyNodes, dtype=int)
bodyBCInds[ whichDir==0 ] = v1Indices[ bodyIndices[whichDir==0] ]
bodyBCInds[ whichDir==1 ] = v2Indices[ bodyIndices[whichDir==1] ]
bodyBCInds[ whichDir==2 ] = v3Indices[ bodyIndices[whichDir==2] ]



# get Finite Element operators
GradGradBuilder = MS.makeStiffnessMatrix()
Mixed1Builder, Mixed2Builder = MS.makeMixedMatrices()

MM = GradGradBuilder.getFullSparse().tolil()
M1 = Mixed1Builder.getFullSparse().tolil()
M2 = Mixed2Builder.getFullSparse().tolil()
M3 = Mixed2Builder.getFullSparse().tolil()

# Zero out equaitons for Dirichlet conditions at infinity
for i in range(NUM_VARS):
	MM[outerIndices+i*MS.nNodes,:] = 0
	M1[outerIndices+i*MS.nNodes,:] = 0
	M2[outerIndices+i*MS.nNodes,:] = 0
	M3[outerIndices+i*MS.nNodes,:] = 0

# Zero out equaitons for no penetration conditions at object
MM[bodyBCInds,:] = 0
M1[bodyBCInds,:] = 0
M2[bodyBCInds,:] = 0
M3[bodyBCInds,:] = 0

MM = MM.tocsr()
M1 = M1.tocsr()
M2 = M2.tocsr()
M3 = M3.tocsr()



# Set initial guess
U = np.ones(NUM_VARS*MS.nNodes)
U[v1Indices] = Vinf[0]
U[v2Indices] = Vinf[1]
U[v3Indices] = Vinf[2]
U[enIndices] = en_inf


# Printing inital Solution
MS.printResults(U[rhoIndices], U[v1Indices], U[v2Indices], U[v3Indices], U[enIndices], 0)
printTag = 1


def adjustBodyVelocity(U,it):
	for i in range(nBodyNodes):
		v = np.array([ U[v1Indices[bodyIndices[i]]], U[v2Indices[bodyIndices[i]]], U[v3Indices[bodyIndices[i]]] ])
		amt = v.dot(MS.bodyNodeNormals[i,:])
		
		v -= (1/(numIts-it+1))*amt*MS.bodyNodeNormals[i,:]
		
		U[v1Indices[bodyIndices[i]]] = v[0]
		U[v2Indices[bodyIndices[i]]] = v[1]
		U[v3Indices[bodyIndices[i]]] = v[2]
		



print("Beginning Main Loop", flush=True)
#for it in range(1,numIts+1):
for it in range(1,numIts+1):
	
	adjustBodyVelocity(U,it)	
	
	for subIt in range(numSubIts):
		# Compute flux functions at each node
		F1, F2, F3 = FF.F123(U[rhoIndices], U[v1Indices], U[v2Indices], U[v3Indices], U[enIndices])
		# Compute Jacobian of each flux function
		DF1DU, DF2DU = FF.df123dU(U[rhoIndices], U[v1Indices], U[v2Indices], U[v3Indices], U[enIndices])

		# Compute Residual
		F = (M1 @ F1) + (M2 @ F2) + (M3 @ F3) - visc*( MM @ U )
		err = np.linalg.norm(F)
		if err > 1e9 :
			print("Blew Up!")
			break
		if err < 1e-9 :
			print("Converged")
			break
	
		# Compute Jacobian of Residual
		DFDU = (M1 @ DF1DU) + (M2 @ DF2DU) + (M3 @ DF3DU) - visc*MM

		# Enforce Boundary Conditions
		DFDU = DFDU.tolil()
		for i in range(NUM_VARS):
			for j in range(len(outerIndices)):
				DFDU[ outerIndices[j]+i*MS.nNodes, outerIndices[j]+i*MS.nNodes ] = 1

		# Enforce No Penetration
		for i in range(nBodyNodes):
			DFDU[ bodyBCInds[i], v1Indices[bodyIndices[i]] ] = MS.bodyNodeNormals[i,0]
			DFDU[ bodyBCInds[i], v2Indices[bodyIndices[i]] ] = MS.bodyNodeNormals[i,1]
			DFDU[ bodyBCInds[i], v3Indices[bodyIndices[i]] ] = MS.bodyNodeNormals[i,2]
	
		DFDU = DFDU.tocsr()

		del_U = spsolve(DFDU, -F)
		U += del_U

	# Printing Solution
	MS.printResults(U[rhoIndices], U[v1Indices], U[v2Indices],  U[v3Indices], U[enIndices], printTag)
	printTag += 1
	
	if err > 1e9 or err < 1e-9 :
		break

print(np.linalg.norm(F))



