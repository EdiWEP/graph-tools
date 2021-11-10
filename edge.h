#pragma once
#include<vector>

using std::vector;

// Auxiliary struct for holding edge information
struct Edge {                                       
        int source, destination, cost;          
        Edge(int source, int dest, int cost = 0): source(source), destination(dest), cost(cost){}
        // Returns edge with source and destination swapped
        Edge Flip() {return Edge(destination, source, cost);}   
};
