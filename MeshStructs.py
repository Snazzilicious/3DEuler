
import numpy as np

DIMENSION = 3

nNodes = -1
nElem = -1

ElemVertInds = None
VertCoords = None
groupNames = None
groupMembers = None

Volumes = None
c0s = None
c1s = None
c2s = None
c3s = None

basisCoeffs = None #[element, xyzc, vertex]

bodyNodeNormals = None



""" Mesh Loading Routines """


def parseMesh(filename):


	global nNodes
	global nElem
	global ElemVertInds
	global VertCoords
	global Areas
	global c0s
	global c1s
	global c2s
	global c3s
	global basisCoeffs
	global groupNames
	global groupMembers

	meshFile = open(filename,'r')
	
	#chop first line
	line = meshFile.readline()

	#get how many elements there are
	line = meshFile.readline().split()
	nElem = int(line[-1])

	#get indices of element vertices
	ElemVertInds = np.zeros([nElem,DIMENSION+1]).astype(int)
	for i in range(nElem):
		line = meshFile.readline().split()
		for j in range(DIMENSION+1):
			ElemVertInds[i,j] = int(line[j+1])


	#get how many vertices there are
	line = meshFile.readline().split()
	nNodes = int(line[-1])

	#get indices of cell vertices
	VertCoords = np.zeros([nNodes,DIMENSION])
	for i in range(nNodes):
		line = meshFile.readline().split()
		for j in range(DIMENSION):
			VertCoords[i,j] = float(line[j])


	# remove unconnected nodes
	fromInd = []
	for i in range(nNodes):
		if i in ElemVertInds:
			fromInd.append(i)
			ElemVertInds[ElemVertInds == i] = len(fromInd)-1
	VertCoords = VertCoords[fromInd,:]
	nNodes = VertCoords.shape[0]
	
	# PRINT VTK MESHFILE
	makeGridFile(ElemVertInds,VertCoords,"meshFile.vtk")


	#get how many groups there are
	line = meshFile.readline().split()
	nGroups = int(line[-1])


	groupSizes = np.zeros(nGroups).astype(int)
	groupNames = ["" for _ in range(nGroups)]
	groupMembers = [[] for _ in range(nGroups)]
# TODO make sure this works
	for i in range(nGroups):
		#get group name
		line = meshFile.readline().split()
		groupNames[i] = line[-1]
		#get group size
		line = meshFile.readline().split()
		groupSizes[i] = int(line[-1])
		#get group data
		for j in range(groupSizes[i]):
			line = meshFile.readline().split()
			for k in range(DIMENSION):
				ind = int(line[-(1+k)])
				if ind in fromInd: # must adjust for deleted nodes
					ind = fromInd.index(ind)
					
					if ind not in groupMembers[i]:
						groupMembers[i].append(ind)
		

	meshFile.close()
	
	# Computing element properties
	c0s = VertCoords[ ElemVertInds[:,0], : ]
	c1s = VertCoords[ ElemVertInds[:,1], : ]
	c2s = VertCoords[ ElemVertInds[:,2], : ]
	c3s = VertCoords[ ElemVertInds[:,3], : ]
	
	Volumes = getElemVolumes(c0s, c1s, c2s, c3s)
	
	basisCoeffs = get3DBasisCoeffs(c0s, c1s, c2s, c3s) #[element, xyc, vertex]
	
	getBodyOutwardNormals()
# END PARSEMESH


def makeGridFile(vertexIndices, vertexCoords, filename):
	numVertices = vertexCoords.shape[0]
	numCells = vertexIndices.shape[0]
	
	
	cellData = 4*np.ones([numCells,5]).astype(int)
	cellData[:,1:] = vertexIndices[:,:]
	
	f = open(filename,'w')
	
	#header
	f.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n")
	
	#print coordinates of vertices
	f.write("POINTS " + str(numVertices) + " float\n")
	for i in range(numVertices):
		f.write(str(vertexCoords[i,0]) + " " + str(vertexCoords[i,1]) + " " + str(vertexCoords[i,2]) + "\n" )

	
	#print indices of vertices
	f.write("CELLS " + str(numCells) + " " + str(5*numCells) + "\n")
	for i in range(numCells):
		f.write(str(cellData[i,0]) + " " + str(cellData[i,1]) + " " + str(cellData[i,2]) \
				+ " " + str(cellData[i,3]) + " " + str(cellData[i,4]) + "\n" )
	
	#cell types
	f.write("CELL_TYPES " + str(numCells) + "\n")
	for i in range(numCells):
		f.write(str(10)+"\n")
	
	f.close()
#END OF makeGridFile


def printResults(rho,v1,v2,v3,en, tag):
	from FluxFunctions import pressure
	
	P = pressure(rho,en)
	
	
	filename = "results.vtk." + str(tag)
	f = open(filename,'w')
	
	
	f.write( "POINT_DATA " + str(nNodes) + "\n" )
	
	#Scalar values
	f.write("SCALARS rho float\nLOOKUP_TABLE default\n")
	for i in range(nNodes):
		f.write( str(rho[i]) + "\n" )


	f.write("SCALARS e float\nLOOKUP_TABLE default\n")
	for i in range(nNodes):
		f.write(str(en[i]) + "\n")


	f.write("SCALARS T float\nLOOKUP_TABLE default\n")
	for i in range(nNodes):
		f.write(str(en[i]/718.0) + "\n")

	
	f.write("SCALARS P float\nLOOKUP_TABLE default\n")
	for i in range(nNodes):
		f.write(str(P[i]) + "\n")

	
	#Vector values
	f.write("VECTORS V float\n")
	for i in range(nNodes):
	
		f.write(str(v1[i]) + " " + str(v2[i]) + " " + str(v3[i]) + "\n")
	
	f.close()


""" Volume of Elements """

def tetVolume(verts):
	# verts is 4 by 3
	
	v1 = verts[0,:] - verts[1,:]
	v2 = verts[2,:] - verts[1,:]
	v3 = verts[3,:] - verts[1,:]
	
	return np.abs( v1[0]*(v2[1]*v3[2]-v2[2]*v3[1]) - v1[1]*(v2[0]*v3[2]-v2[2]*v3[0]) + v1[2]*(v2[0]*v3[1]-v2[1]*v3[0]) )/6.0

def getElemVolumes(c0s, c1s, c2s, c3s):
	V1s = c0s - c1s
	V2s = c2s - c1s
	V3s = c3s - c1s
	
	return np.abs( V1[:,0]*(V2[:,1]*V3[:,2]-V2[:,2]*V3[:,1]) - V1[:,1]*(V2[:,0]*V3[:,2]-V2[:,2]*V3[:,0]) + V1[:,2]*(V2[:,0]*V3[:,1]-V2[:,1]*V3[:,0]) )/6.0

def get3DBasisCoeffs(c0s, c1s, c2s, c3s):
	# Computing the coefficients for the basis functions
	allMats = np.zeros([nElem,DIMENSION+1,DIMENSION+1])
	allMats[:,:,0] = np.column_stack((c0s[:,0],c1s[:,0],c2s[:,0],c3s[:,0]))
	allMats[:,:,1] = np.column_stack((c0s[:,1],c1s[:,1],c2s[:,1],c3s[:,1]))
	allMats[:,:,2] = np.column_stack((c0s[:,2],c1s[:,2],c2s[:,2],c3s[:,2]))
	allMats[:,:,3] = np.ones([nElem,DIMENSION+1])
	
	rhs = np.zeros([nElem,DIMENSION+1,DIMENSION+1])
	for i in range(DIMENSION+1):
		rhs[:,i,i] = 1.0
	return np.linalg.solve(allMats, rhs) #[element, xyzc, vertex]

""" End of Mesh Loading Routines """



""" Sparse Matrix Construction Routines and Structures """


class spMatBuilder:
	def __init__(self, N):
		self.N = N
		self.dat=[]
		self.rowPtr=np.zeros(N+1).astype(int)
		self.colInds = []
		
	def addEntry(self,i,j,val):
		
		if self.rowPtr[i] == self.rowPtr[i+1]:
			self.colInds.insert(self.rowPtr[i],j)
			self.dat.insert(self.rowPtr[i],val)
			self.rowPtr[i+1:] += 1
		else:
		
			done = False
			# See if the indices already exits
			for col in range(self.rowPtr[i],self.rowPtr[i+1]):
				if self.colInds[col] == j:
					self.dat[col] += val
					done = True
			if not done:
				# see if should be the last column
				if j > self.colInds[self.rowPtr[i+1]-1]:
					self.colInds.insert(self.rowPtr[i+1],j)
					self.dat.insert(self.rowPtr[i+1],val)
				else:
					for col in range(self.rowPtr[i], self.rowPtr[i+1]):
						if j < self.colInds[col]:
							self.colInds.insert(col,j)
							self.dat.insert(col,val)
							break
				self.rowPtr[i+1:] += 1
		
	
	def getDense(self):
		res = np.zeros([self.N,self.N])
		for row in range(self.N):
			for j in range(self.rowPtr[row], self.rowPtr[row+1]):
				res[ row, self.colInds[j] ] = self.dat[j]
		return res
	
	def getSparse(self):
		from scipy.sparse import csr_matrix
		return csr_matrix((self.dat,self.colInds,self.rowPtr), shape=(self.N,self.N))
	
	def getFullSparse(self):
		# 4 is b/c 4 variables
		from scipy.sparse import csr_matrix
		nnz = len(self.dat)
		
		fullDat = self.dat*4
		
		fullColInds = np.array(self.colInds*4)
		
		fullRowPtr = np.zeros( 4*self.N + 1 )
		fullRowPtr[:self.N+1] = self.rowPtr[:self.N+1]
		
		for i in range(1,4):
			fullColInds[i*nnz:(i+1)*nnz] += i*self.N
			fullRowPtr[ i*self.N+1 : (i+1)*self.N+1 ] = self.rowPtr[1:] + fullRowPtr[i*self.N]
		
		return csr_matrix((fullDat,fullColInds,fullRowPtr), shape=(4*self.N,4*self.N))



def makeStiffnessMatrix():
	
	builder = spMatBuilder(nNodes)
	
	for elem in range(nElem):
		for i in range(DIMENSION+1):
			for j in range(DIMENSION+1):
				vert1 = ElemVertInds[elem,i]
				vert2 = ElemVertInds[elem,j]
				# TODO consider normalizing gradients to emphasize averaging term, not true Laplacian
				grad1 = basisCoeffs[elem,:DIMENSION,i] 
				grad2 = basisCoeffs[elem,:DIMENSION,j]
				
				val = Volumes[elem]*grad1.dot(grad2)
				builder.addEntry(vert1,vert2,val)
	
	return builder



def makeMixedMatrices():
	
	builder1 = spMatBuilder(nNodes)
	builder2 = spMatBuilder(nNodes)
	builder2 = spMatBuilder(nNodes)
	
	for elem in range(nElem):
		for i in range(DIMENSION+1):
			for j in range(DIMENSION+1):
				vert1 = ElemVertInds[elem,i]
				vert2 = ElemVertInds[elem,j]
				
				grad1 = basisCoeffs[elem,:DIMENSION,i] 
				
				val1 = Volumes[elem]*grad1[0]/4.0
				val2 = Volumes[elem]*grad1[1]/4.0
				val3 = Volumes[elem]*grad1[2]/4.0
				
				builder1.addEntry(vert1,vert2,val1)
				builder2.addEntry(vert1,vert2,val2)
				builder3.addEntry(vert1,vert2,val3)

	# Boundary terms at the surface of the object
	GID = groupNames.index( 'Body' )	
	normal = np.zeros([DIMENSION])
	for elem in range(nElem):
		# see which nodes belong to the body
		bodyNodes = [ ElemVertInds[elem,i] in groupMembers[GID] for i in range(DIMENSION+1) ]
		
		# if a face of the element is a piece of the body
		if np.sum(bodyNodes) == DIMENSION:
			notBodyNodes = [not val for val in bodyNodes]
		
			vert1 = ElemVertInds[elem,bodyNodes][0]
			vert2 = ElemVertInds[elem,bodyNodes][1]
			nodeCoords = VertCoords[ ElemVertInds[elem,bodyNodes], : ]
			
			#TODO compute outward normal






















