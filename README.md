# GraphTools
A C++ class with various methods for operations on graphs, binary heaps, disjoint sets and flow networks


## Classes

- **Graph** -
Used for the majority of all algorithms, this class represents a graph, it can be directed or undirected, weighted or unweighted

- **Heap** -
This class represents a binary heap. It is a template class, and the template type determines the type of the values the nodes will hold, which can also be std::pair

- **DisjointSets** -
A struct that is used for operations on disjoint set data structures, it has the fundamental Root() and Union() methods

- **FlowHandler** - 
An auxiliary class used for operations on flow networks within the Graph class


## Some featured methods of Graph

- MinimumSpanningTree() - Returns a new graph that is the MST of the calling graph
- WeightedDistances() - Returns a vector of distances from one node to all others, it uses Dijkstra's algorithm if there are no negative values, otherwise it uses Bellman-Ford
- UnweightedDistances() - Returns a vector of distances from one node to all others, where distance is the number of edges from one node to another
- AllMinimumDistances() - Returns a vector of vectors of distances between nodes, returned[i][j] represents the distance from node i to node j, uses the Roy-Floyd algorithm
- CriticalConnections() - Returns a vector of all critical edges within the graph
- NumberOfComponents() - Returns the number of connected components the graph has
- TreeDiameter() - Returns the diameter of the graph, should only be used on trees
- CheckHavelHakimi() - Returns true if a graph can be created from the given vector of node degree values, uses an optimized version of the Havel-Hakimi algorithm
- MaximumFlow() - Returns the maximum flow that can pass through the flow network represented by the graph. Must use the FlowHandler that was passed to BuildFlowNetwork()

## For reading input
These methods should be used after instantiating the graph and setting the numberOfNodes and numberOfEdges

- BuildFromAdjacencyList() - Given an input stream, reads the graph's edges assuming an adjacency list format
- BuildFromAdjacencyMatrix() - Given an input stream, reads the graph's edges assuming an adjacency matrix format
- BuildFlowNetwork() - Given an input stream and a FlowHandler instance, assumes an adjacency list-like of a flow network (nodeX, nodeY, capacity, cost)

## For modifying a graph

- AddEdge() - Adds a new edge, can be given an Edge struct object or the source, destination and cost. This changes the numberOfEdges property
- AddNode() - Adds a new node with a new highest index. This changes the numberOfNodes property.
- SetWeighted() - Sets if the graph is weighted through a boolean value. Does not erase edge weight data
- SetWeighted() - Sets if the graph is directed through a boolean value
- SetValue() - Sets the value of a given node(by indeX).
