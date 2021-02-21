#!/bin/bash
#makes vtk files with results from solver
#using the mesh vtk file from the solver

rm -f final*vtk

for f in $(ls results.*); 
	do cat meshFile.vtk > "final$f" ; 
	cat $f >> "final$f"  ; 
done
