#include<iostream>
#include<fstream>
#include<vector>
#include<queue>
#include<stack>
#include<algorithm>

using namespace std;



class Graph {

    public:
    
    struct Edge {                                       // Auxiliary struct for holding edge information
        int source; int destination; int cost;          // Source is the source node,
                                                        // Destinaiton is the node the edge points towards, cost is the edge's cost
        operator int() { return cost; } 
        Edge(int source, int dest, int cost = 0): source(source), destination(dest), cost(cost){}

        Edge flip() {return Edge(destination, source, cost);}   // Returns edge with source and destination swapped
    };

    private:

        bool directed;
        bool weighted;

        int numberOfNodes;
        int numberOfEdges;

        vector < int > values;                              // The value of each node 
        vector < vector < Edge > > adjacencyList;           // adjacencyList[X] holds the information of each edge that X has
        
        void BFS(vector<int>& visitedNodes, vector<int>& distances, int startIndex = 0);
        void TreeBuilderBFS(int startIndex, Graph& treeGraph, vector<bool>& visitedNodes);

        void DFS(vector<int>& visitedNodes, int marker = 1, int nodeIndex = 0);
        void StronglyConnectedDFS(int currentNode, vector<vector<int>>& scc, vector<int>& order, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& onStack, int& counter);
        void BiconnectedDFS(int currentNode, int parent, int currentDepth, vector< vector <int>>& bcc, vector<int>& depth, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& visited);
        void CriticalEdgesDFS(int currentNode, int parent, vector <Edge>& cc, vector<int>& depth, vector<int>& lowest, int& counter);
        void TopologicalDFS(int currentNode, stack<int>& orderStack, vector<bool>& visitedNodes);
        void TreeBuilderDFS(int currentNode, int treeCurrentNode, Graph& treeGraph, vector<bool>& visitedNodes);

    public:

        int NumberOfComponents();
        vector<Edge> ListOfEdges();
        vector<int> UnweightedDistances(int startIndex = 0);
        vector<int> TopologicalSort();
        vector<vector <int>> StronglyConnectedComponents();
        vector<vector <int>> BiconnectedComponents();
        vector<Edge> CriticalConnections();

        Graph DFSTree(int startIndex);
        Graph BFSTree(int startIndex);
        Graph DFSTrees();
        
        static bool CheckHavelHakimi(vector<int>& degrees);

        void BuildFromAdjacencyList(istream& inputStream);
        void BuildFromAdjacencyMatrix(istream& inputStream);

        void AddEdge(int source, int dest, int cost = 0) {

            Edge newEdge(source, dest, cost);

            adjacencyList[source].push_back(newEdge);

            if(!this->directed) {

                adjacencyList[dest].push_back(newEdge.flip());
            }

            ++numberOfEdges;
        }

        void AddEdge(Edge newEdge) {

            adjacencyList[newEdge.source].push_back(newEdge);

            if(!this->directed) {

                adjacencyList[newEdge.destination].push_back(newEdge.flip());
            }

            ++numberOfEdges;
        }

        void AddNode(int value = 0) {

            vector<Edge> tempVector;
            adjacencyList.push_back(tempVector);
    
            values.push_back(value);

            ++numberOfNodes;
        }




#pragma region GraphConstructors

        Graph(bool directed = false, bool weighted = false) {

            numberOfNodes = 0;
            numberOfEdges = 0;
            this->directed = directed;
            this->weighted = weighted; 
        }

        Graph(int numberOfNodes, int numberOfEdges, bool directed = false, bool weighted = false) {
            
            this->numberOfEdges = numberOfEdges;
            this->directed = directed;
            this->weighted = weighted; 

            for(int i = 0; i < numberOfNodes; ++i) {
                
                AddNode();
            }

        }

#pragma endregion

#pragma region GraphGetSet

        void SetValue(int node, int value) {

            try {
                values[node] = value;
            }
            catch(...) {
                throw "Node doesn't exist";
            }
        }

        int GetValue(int node) {
            
            try {
                return values[node];
            }
            catch(...) {
                throw "Node doesn't exist";
            }
            
        }

        void SetDirected(bool directed) {
            this->directed = directed;
        }

        bool IsDirected() {
            return directed;
        }

        void SetWeighted(bool weighted) {
            this->weighted = weighted;
        }

        bool IsWeighted() {
            return weighted;
        }

        void SetNumberOfNodes(int number) {
            numberOfNodes = number;
        }
        
        int GetNumberOfNodes() {
            return numberOfNodes;
        }

        void SetNumberOfEdges(int number) {
            numberOfEdges = number;
        }

        int GetNumberOfEdges() {
            return numberOfEdges;
        }

#pragma endregion

};

#pragma region GraphPrivateMethods


void Graph::TreeBuilderBFS(int startIndex, Graph& treeGraph, vector<bool>& visitedNodes) {
                                                                        // BFS that also builds a new graph(the BFS tree)
    queue<int> nodeQueue;
    treeGraph.AddNode();
    int treeCurrentNode = 0;        // Index of the current node in the new graph

    nodeQueue.push(startIndex);

    while(!nodeQueue.empty()) {

        int topNode = nodeQueue.front();

        visitedNodes[topNode] = true;

        for (auto edge : adjacencyList[topNode]) {

            int neighbor = edge.destination;
             
            if(!visitedNodes[neighbor]) {

                treeGraph.AddNode();
                treeGraph.AddEdge(treeCurrentNode, treeGraph.numberOfNodes-1);

                nodeQueue.push(neighbor);
                visitedNodes[neighbor] = true;
            }
        }

        treeCurrentNode += 1;       // The next node the search will process has this index in the tree
        nodeQueue.pop();
    }

}

void Graph::TreeBuilderDFS(int currentNode, int treeCurrentNode, Graph& treeGraph, vector<bool>& visitedNodes) {
                                                                        // DFS that also builds a new graph(the DFS tree)
    visitedNodes[currentNode] = true;

    for(auto edge : adjacencyList[currentNode]) {

        int child = edge.destination;

        if(!visitedNodes[child]) {

            treeGraph.AddNode();
            treeGraph.AddEdge(treeCurrentNode, treeGraph.numberOfNodes-1);

            TreeBuilderDFS(child, treeGraph.numberOfNodes-1, treeGraph, visitedNodes);
        }
    }
}

void Graph::BFS(vector<int>& visitedNodes, vector<int>& distances, int startIndex /*= 0*/) {
                                                                        // Breadth-first search that sets visited nodes and distance to each node from node with index startIndex
    queue<int> nodeQueue;           // Queue of nodes that haven't had their edges processed yet

    nodeQueue.push(startIndex);
    distances[startIndex] = 0;      // Distance to starting node

    while(!nodeQueue.empty()) {

        int topNode = nodeQueue.front();

        visitedNodes[topNode] = 1;

        for(auto edge : adjacencyList[topNode]) {

            int neighbor = edge.destination;

            if(!visitedNodes[neighbor]) {
                
                nodeQueue.push(neighbor);
                distances[neighbor] = 1 + distances[topNode];
                visitedNodes[neighbor] = 1;
            }
        }

        nodeQueue.pop();
    }

}

void Graph::DFS(vector<int>& visitedNodes, int marker /*= 1*/, int nodeIndex /*= 0*/) {
                                                                                        // Recursive depth-first search, sets visited positions in visitedNodes with marker for counting components
    visitedNodes[nodeIndex] = marker;

    for(auto edge : adjacencyList[nodeIndex]) {

        int neighbor = edge.destination;

        if(!visitedNodes[neighbor]) {
            DFS(visitedNodes, marker, neighbor);
        }
    }

}

void Graph::StronglyConnectedDFS(int currentNode, vector<vector<int>>& scc, vector<int>& order, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& onStack, int& counter) {

    order[currentNode] = counter;
    lowest[currentNode] = counter;

    ++counter;

    nodeStack.push(currentNode);
    onStack[currentNode] = true;


    for(auto edge : adjacencyList[currentNode]) {

        int neighbor = edge.destination;

        if(order[neighbor] < 0) {           // If node hasn't been visited yet

            StronglyConnectedDFS(neighbor, scc, order, lowest, nodeStack, onStack, counter);

            lowest[currentNode] = min(lowest[currentNode], lowest[neighbor]);               // Set the SCC index of each node on the recursion return path
        }
        else {
            if (onStack[neighbor]) {
                                                                                            // If neighbor isn't on the stack, then neighbor is part of a different, previously discovered SCC
                lowest[currentNode] = min(lowest[currentNode], order[neighbor]);
            }
        }
    }

    if(lowest[currentNode] == order[currentNode]) {                                         // If lowest[X] = order[X] then X has no back-edge and is the root of its SCC
                                                                                            // The stack must be popped up to said root, as we have found a complete SCC                                                                                      
        vector<int> currentScc;

        int stackTop;

        do {

            stackTop = nodeStack.top();
            
            nodeStack.pop();
            onStack[stackTop] = false;

            currentScc.push_back(stackTop);

        } while (currentNode != stackTop);

        scc.push_back(currentScc);
    }
}

void Graph::CriticalEdgesDFS(int currentNode, int parent, vector <Edge>& cc, vector<int>& order, vector<int>& lowest, int& counter) {

    order[currentNode] = counter;
    lowest[currentNode] = counter;

    ++counter;

    for(auto edge : adjacencyList[currentNode]) {

        int child = edge.destination;

        if(order[child] < 0) {

            CriticalEdgesDFS(child, currentNode, cc, order, lowest, counter);

            lowest[currentNode] = min(lowest[currentNode], lowest[child]);      // Update lowest after subtree is finished

            if (lowest[child] > order[currentNode]) {
                                                            // We can't reach the current node through any path that doesn't include the current edge
                                                            // So current edge is critical
                cc.push_back(edge);            
            }
        }
        else if (child != parent || directed) {                         // Found visited node, so a back-edge
                                                            // The if condition is necessary so we don't go back through the edge we just used
            lowest[currentNode] = min(lowest[currentNode], order[child]);
        }
    }
}

void Graph::BiconnectedDFS(int currentNode, int parent, int currentDepth, vector< vector <int>>& bcc, vector<int>& depth, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& visited) {

    depth[currentNode] =  currentDepth;
    lowest[currentNode] = currentDepth;
    visited[currentNode] = true;
    nodeStack.push(currentNode);

    for(auto edge : adjacencyList[currentNode]) {

        int child = edge.destination;

        if (child != parent) {          // Check is required to prevent loops
                                    
            if(visited[child]) {                    

                lowest[currentNode] = min(lowest[currentNode], depth[child]);               // Back-edge is found
            }

            else {

                BiconnectedDFS(child, currentNode, currentDepth +1, bcc, depth, lowest, nodeStack, visited);

                lowest[currentNode] = min(lowest[currentNode], lowest[child]);              // Update with lowest depth descendants can reach

                if(depth[currentNode] <= lowest[child]) {                                    // If true, then child can't reach above current depth, so current node separates two BCCs

                    
                    vector < int > currentBcc;
                    currentBcc.push_back(currentNode);                  // Current node is part of both BCCs that it separates

                    int stackTop;

                    do {

                        stackTop = nodeStack.top();
                        nodeStack.pop();
                        currentBcc.push_back(stackTop);
                    } while ( stackTop != child );

                    bcc.push_back(currentBcc);
                }
            }
        }
    }
}

void Graph::TopologicalDFS(int currentNode, stack<int>& orderStack, vector<bool>& visitedNodes) {

    visitedNodes[currentNode] = true;

    for(auto edge : adjacencyList[currentNode]) {

        int node = edge.destination;

        if(!visitedNodes[node]) {

            TopologicalDFS(node, orderStack, visitedNodes);
        }
    }
    
    orderStack.push(currentNode);           // The stack is formed in the return order of the DFS
}

#pragma endregion

#pragma region GraphPublicMethods

Graph Graph::DFSTree(int startIndex) {
                                            // Returns a graph object of the DFS tree of node with startIndex index
                                            // Node of index startIndex will be the node of index 0 in the new graph
    Graph treeGraph(1, 0, directed, weighted);        // Initialize empty graph with one node
    vector<bool> visitedNodes;

    for(int i = 0; i < numberOfNodes; ++i) {

        visitedNodes.push_back(false);
    }

    TreeBuilderDFS(startIndex, 0, treeGraph, visitedNodes);
    
    return treeGraph;
}

Graph Graph::DFSTrees() {
                                            // Returns a graph object of the DFS tree of node with startIndex index
                                            // Node of index startIndex will be the node of index 0 in the new graph
    Graph treeGraph(0, 0, directed, weighted);        // Initialize empty graph
    vector<bool> visitedNodes;

    for(int i = 0; i < numberOfNodes; ++i) {

        visitedNodes.push_back(false);
    }

    for(int node = 0; node < numberOfNodes; ++node) {
        
        if(!visitedNodes[node]) {

            treeGraph.AddNode();
            TreeBuilderDFS(node, treeGraph.numberOfNodes-1, treeGraph, visitedNodes);
        }
    }

    return treeGraph;
}

Graph Graph::BFSTree(int startIndex) {

    Graph treeGraph(0, 0, directed, weighted);
    vector<bool> visitedNodes;

    for(int i = 0; i < numberOfNodes; ++i) {

        visitedNodes.push_back(false);
    }

    TreeBuilderBFS(startIndex, treeGraph, visitedNodes);

    return treeGraph;
}

vector<int> Graph::TopologicalSort() {
                                            // Returns the list of nodes in topologically sorted order
                                            // Should only be applied on directed graphs
    vector<int> sortedNodes;

    stack<int> orderStack;                  // By using a stack, the elements in pop order will form the topologically sorted list
    vector<int> depth;
    vector<bool> visited;

    for(int i = 0; i < numberOfNodes; ++i) {

        visited.push_back(false);    
    }

    for(int i = 0; i < numberOfNodes; ++i) {

        if(!visited[i]) { 
            TopologicalDFS(i, orderStack, visited);
        }
    }

    for(int i = 0; i < numberOfNodes; ++i) {

        sortedNodes.push_back(orderStack.top());
        orderStack.pop();
    }

    return sortedNodes;

}

bool Graph::CheckHavelHakimi(vector<int>& degrees) {
                                                                            // Receives a list of node degrees and returns true if a corelated graph can exist through the Havel-Hakimi algorithm
    int numberOfNodes = degrees.size();

    int sum = 0; 

    for(int degree : degrees) {

        sum += degree;

        if (degree >= numberOfNodes) {
                                    // If degree is > n-1 then there are not enough nodes, therefore a graph doesn't exist
            return false; 
        }
    }

    if(sum % 2) {
                            // If sum of degrees is odd then a graph doesn't exist
        return false; 
    }

    sort(degrees.begin(), degrees.end());           // Sort vector of degrees, then iterate through it in descending order

    --numberOfNodes; 

    while(sum > 0) {

        int currentNode = degrees[numberOfNodes];

        for(int i = numberOfNodes; i > numberOfNodes - currentNode - 1; --i ) {
            
            --degrees[i];                       // Decrement the previous max(degrees) values
            if(degrees[i] < 0) {
                                               // If any number in the vector falls below 0, then a graph doesn't exist
                return false;
            }
        }

        --numberOfNodes;        // "Eliminating" the node we resolved the degrees of
        sum -= 2*currentNode;
       
        degrees.pop_back();
        sort(degrees.begin(), degrees.end());   // Bubble sort might be faster here
    }
    
    return true;            // If this point is reached then the vector is now [0, 0, .. 0] and a graph exists for the given set of degrees          
}

vector<Graph::Edge> Graph::CriticalConnections() {
                                                         // Tested on leetcode(Critical Connections problem)  
    vector<Edge> cc;                    // The list of critical edges

    vector<int> order;                  // Node X is the order[x]-th node to be found during DFS
    vector<int> lowest;                 // lowest[X] is the minimum order[Y], where node Y is connected to node X
     
    int counter = 0;

    for(int i = 0; i < numberOfNodes; ++i) {

        order.push_back(-1);            // -1 order means node hasn't been visited yet
        lowest.push_back(-1);
    }

    CriticalEdgesDFS(0, -1, cc, order, lowest, counter);  // Root node has no parent

    return cc;
} 

vector<int> Graph::UnweightedDistances(int startIndex /*= 0*/) {
                                                                        // Wrapper Method for calculating unweighted distances to each node form startIndex through BFS
                                                                        // Saves distances in distances parameter(distances vector should be empty when calling this method)
    
    vector<int> visited;
    vector<int> distances;

    
    for(int i = 0; i < numberOfNodes; ++i) {
        visited.push_back(0);
        distances.push_back(-1);                // -1 means there is no path to a certain node
    }

    BFS(visited, distances, startIndex);

    return distances;
}

int Graph::NumberOfComponents() {
                                        // Computes number of components in undirected graph 
    int numberOfComponents = 0;

    vector<int> visited;
    for(int i = 0; i < numberOfNodes; ++i) {
        visited.push_back(0);
    }

    for(int i = 0; i < numberOfNodes; ++i) {

        if (!visited[i]) {
            ++numberOfComponents;
            DFS(visited, numberOfComponents, i);
        }
    }

    return numberOfComponents;
}

vector< vector <int> > Graph::BiconnectedComponents() {
                                                                    // Returns the list of biconnected components in graph

    vector< vector<int> > bcc;      // The list of BCCs to be returned

    stack<int> nodeStack;           // The stack of visited nodes
    
    vector<int> depth;              // Holds the depth of each node in the DFS tree
    vector<int> lowest;             // Holds the lowest depth a node can reach through its descendants
    vector<bool> visited;           // For marking visited nodes

    for(int i = 0; i < numberOfNodes; ++i) {
        depth.push_back(-1);        // Initialization
        lowest.push_back(-1);   
        visited.push_back(false);
    }

    BiconnectedDFS(0, -1, 0, bcc, depth, lowest, nodeStack, visited);       // Root node (0) has no parent

    return bcc;
}

vector< vector <int> > Graph::StronglyConnectedComponents() {
                                                                    // Computes the number of strongly connected components in directed graph through Tarjan's algorithm
    vector< vector <int> > scc;         // The list of strongly connected components to be returned

    stack<int> nodeStack;               // Stack to be used in tarjan's algorithm

    vector<int> order;                  // Node X is the order[x]-th node to be found during DFS
    vector<int> lowest;                 // lowest[X] is the minimum order[Y], where node Y is connected to node X
    vector<bool> onStack;               // onStack[X] is true if node X is currently on the stack 
    
    int counter = 0;                    // Counter for the order vector

    for(int i = 0; i < numberOfNodes; ++i) {
        order.push_back(-1);            // -1 means node hasn't been visited
        lowest.push_back(-1);           
        onStack.push_back(false);
    }


    for(int node = 0; node < numberOfNodes; ++node) {

        if (order[node] < 0) {

            StronglyConnectedDFS(node, scc, order, lowest, nodeStack, onStack, counter);
        }
    }

    return scc;
}

void Graph::BuildFromAdjacencyMatrix(istream& inputStream) {        // Sets edges between nodes by reading an adjacency matrix from inputStream
                                                                    // Should only be used if numberOfEdges has already been set 

    int matrixValue;             // If matrixValue is 0, there is no edge, if matrixValue != 0, then there is an edge of cost matrixValue

    for(int i = 0; i < numberOfNodes; ++i) {
        for (int j = 0; j < numberOfNodes; ++j) {

            inputStream >> matrixValue;

            if(matrixValue) {
                
                if(!weighted) {
                    matrixValue = 0;
                }

                Edge newEdge(i, j, matrixValue);

                adjacencyList[i].push_back( newEdge );

                if(!directed) {    
                    adjacencyList[j].push_back( newEdge.flip() );
                }
               
            }
        } 
    }
}

void Graph::BuildFromAdjacencyList(istream& inputStream) {          // Sets edges between nodes by reading adjancency list pairs from inputStream
                                                                    // Should only be used if numberOfEdges has already been set 
                                                                    
    int node1, node2, cost;

    for(int i = 0; i < numberOfEdges; ++i) {

        inputStream >> node1 >> node2;
 
        if(weighted) {
            inputStream >> cost;
        }
        else {
            cost = 0;
        }
        
        Edge newEdge(node1, node2, cost);

        adjacencyList[node1].push_back( newEdge );
      
        if(!directed) {
                adjacencyList[node2].push_back( newEdge.flip() );
        }
    }   
}
