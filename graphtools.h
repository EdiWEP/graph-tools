#include<iostream>
#include<fstream>
#include<vector>
#include<queue>
#include<stack>
#include<algorithm>

using namespace std;



class Graph {

    private:

        bool directed;

        int numberOfNodes;
        int numberOfEdges;

        vector < vector < int > > adjacencyList;

        void AddNode();
        void AddEdge(int node1, int node2);

        
        void DFS(vector<int>& visitedNodes, int marker = 1, int nodeIndex = 0);
        void BFS(vector<int>& visitedNodes, vector<int>& distances, int startIndex = 0);
        void TarjanDFS(int currentNode, vector<vector<int>>& ssc, vector<int>& order, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& onStack, int& counter);
        void BiconnectedDFS(int currentNode, int parent, int currentDepth, vector< vector <int>>& bcc, vector<int>& depth, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& visited);
        void CriticalEdgesDFS(int currentnode, int parent, vector < pair <int, int>>& cc, vector<int>& depth, vector<int>& lowest, int& counter);
        void TopologicalDFS(int currentNode, stack<int>& orderStack, vector<bool>& visitedNodes);
        void TreeBuilderDFS(int currentNode, int treeCurrentNode, Graph& treeGraph, vector<bool>& visitedNodes);

    public:

        int NumberOfComponents();
        vector<int> UnweightedDistances(int startIndex = 0);
        vector<int> TopologicalSort();
        vector<vector <int>> StronglyConnectedComponents();
        vector<vector <int>> BiconnectedComponents();
        vector<pair <int, int>> CriticalConnections();

        Graph DFSTree(int startIndex);
        Graph BFSTree(int startIndex);
        Graph DFSTrees();
        
        static bool CheckHavelHakimi(vector<int>& degrees);

        void BuildFromAdjacencyList(istream& inputStream);


#pragma region GraphConstructors

        Graph(bool directed = false) {

            numberOfNodes = 0;
            numberOfEdges = 0;
            this->directed = directed; 
        }

        Graph(int numberOfNodes, int numberOfEdges, bool directed = false) {
            
            this->numberOfNodes = numberOfNodes;
            this->numberOfEdges = numberOfEdges;
            this->directed = directed;

            for(int i = 0; i < numberOfNodes; ++i) {
                vector<int> tempVector;
                adjacencyList.push_back(tempVector);
            }
        }

#pragma endregion

#pragma region GraphGetSet


        void SetDirected(bool directed) {
            this->directed = directed;
        }

        bool IsDirected() {
            return directed;
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

void Graph::AddNode() {

    vector<int> tempVector;
    adjacencyList.push_back(tempVector);
    
    ++numberOfNodes;
}

void Graph::AddEdge(int node1, int node2) {

    adjacencyList[node1].push_back(node2);

    if(!this->directed) {

        adjacencyList[node2].push_back(node1);
    }

    ++numberOfEdges;
}

void Graph::TreeBuilderDFS(int currentNode, int treeCurrentNode, Graph& treeGraph, vector<bool>& visitedNodes) {

    visitedNodes[currentNode] = true;

    for(int child: adjacencyList[currentNode]) {

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

        for(int neighbor : adjacencyList[topNode]) {
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

    for(int neighborIndex : adjacencyList[nodeIndex]) {

        if(!visitedNodes[neighborIndex]) {
            DFS(visitedNodes, marker, neighborIndex);
        }
    }

}

void Graph::TarjanDFS(int currentNode, vector<vector<int>>& ssc, vector<int>& order, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& onStack, int& counter) {

    order[currentNode] = counter;
    lowest[currentNode] = counter;

    ++counter;

    nodeStack.push(currentNode);
    onStack[currentNode] = true;


    for(int neighbor: adjacencyList[currentNode]) {

        if(order[neighbor] < 0) {           // If node hasn't been visited yet

            TarjanDFS(neighbor, ssc, order, lowest, nodeStack, onStack, counter);

            lowest[currentNode] = min(lowest[currentNode], lowest[neighbor]);               // Set the SSC index of each node on the recursion return path
        }
        else {
            if (onStack[neighbor]) {
                                                                                            // If neighbor isn't on the stack, then neighbor is part of a different, previously discovered SSC
                lowest[currentNode] = min(lowest[currentNode], order[neighbor]);
            }
        }
    }

    if(lowest[currentNode] == order[currentNode]) {                                         // If lowest[X] = order[X] then X has no back-edge and is the root of its SSC
                                                                                            // The stack must be popped up to said root, as we have found a complete SSC                                                                                      
        vector<int> currentSsc;

        int stackTop;

        do {

            stackTop = nodeStack.top();
            
            nodeStack.pop();
            onStack[stackTop] = false;

            currentSsc.push_back(stackTop);

        } while (currentNode != stackTop);

        ssc.push_back(currentSsc);
    }
}

void Graph::CriticalEdgesDFS(int currentNode, int parent, vector < pair <int, int>>& cc, vector<int>& order, vector<int>& lowest, int& counter) {

    order[currentNode] = counter;
    lowest[currentNode] = counter;

    ++counter;

    for(int child : adjacencyList[currentNode]) {

        if(order[child] < 0) {

            CriticalEdgesDFS(child, currentNode, cc, order, lowest, counter);

            lowest[currentNode] = min(lowest[currentNode], lowest[child]);      // Update lowest after subtree is finished

            if (lowest[child] > order[currentNode]) {
                                                            // We can't reach the current node through any path that doesn't include the current edge
                                                            // So current edge is critical
                pair <int, int> edge;
                edge.first = currentNode; edge.second = child;

                cc.push_back(edge);            
            }
        }
        else if (child != parent) {                         // Found visited node, so a back-edge

            lowest[currentNode] = min(lowest[currentNode], order[child]);
        }
    }
}

void Graph::BiconnectedDFS(int currentNode, int parent, int currentDepth, vector< vector <int>>& bcc, vector<int>& depth, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& visited) {

    depth[currentNode] =  currentDepth;
    lowest[currentNode] = currentDepth;
    visited[currentNode] = true;
    nodeStack.push(currentNode);

    for(int child: adjacencyList[currentNode]) {

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

    for(int node: adjacencyList[currentNode]) {

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
    Graph treeGraph(1, 0, directed);        // Initialize empty graph
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
    Graph treeGraph(0, 0, directed);        // Initialize empty graph
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
 return Graph();
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

vector<pair <int,int> > Graph::CriticalConnections() {
                                                         // Tested on leetcode(Critical Connections problem)  
    vector<pair <int, int> > cc;                         // The list of critical edges

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
    vector< vector <int> > ssc;         // The list of strongly connected components to be returned

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

            TarjanDFS(node, ssc, order, lowest, nodeStack, onStack, counter);
        }
    }

    return ssc;
}

void Graph::BuildFromAdjacencyList(istream& inputStream) {           // Sets edges between nodes by reading adjancency list pairs from inputStream

    int node1, node2;

    for(int i = 0; i < numberOfEdges; ++i) {

        inputStream >> node1 >> node2;
 
        adjacencyList[node1].push_back(node2);
      
        if(!directed) {

            adjacencyList[node2].push_back(node1);
        }
    }
    
}

#pragma endregion