// Y-bifurcation: one input, two outputs

// Run with: gmsh bifurcation.geo -1 -format msh2 -o bifurcation.msh

lc = 0.1;

Point(1) = {0, 0, 0, lc};        // Root
Point(2) = {1, 0, 0, lc};        // Branch 1
Point(3) = {2, 1, 0, lc};        // Branch 2
Point(4) = {2, -1, 0, lc};       // Branch 3

Line(1) = {1, 2}; // Root to Branch point
Line(2) = {2, 3}; // Upward branch
Line(3) = {2, 4}; // Downward branch

// Assign a fixed number of elements per line (e.g., 10)
Transfinite Line{1} = 1000;
Transfinite Line{2} = 1000;
Transfinite Line{3} = 1000;

// Save as line elements (1D)
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};

// Physical points (for boundary conditions)
Physical Point("inlet") = {1};
Physical Point("outlet1") = {3};
Physical Point("outlet2") = {4};
