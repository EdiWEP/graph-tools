// infoarena.com/problema/biconex
#include<iostream>
#include<fstream>
#include<vector>
#include<queue>
#include<stack>
#include<algorithm>

using namespace std;


class Node {
                        // Currently has no practical use, this class is here for future versions of this program
    private:

        int value;
        int index;
    
    public:

#pragma region NodeConstructors
        Node() {
            value = 0;
            index = 0;
        }

        Node(int index, int value = 0) {
            this->value = value;
            this->index = index;
        }
#pragma endregion

#pragma region NodeGetSet
        void SetValue(int value) {
            this->value = value;
        }
        void SetIndex(int index) {
            this->index = index;
        }

        int GetValue() {
            return value;
        }
        int GetIndex() {
            return index;
        }
#pragma endregion

};

class Graph {

    private:

        bool directed;

        int numberOfNodes;
        int numberOfEdges;

        vector < Node > nodes;
        vector < vector < int > > adjancencyList;

        void DFS(vector<int>& visitedNodes, int marker = 1, int nodeIndex = 0);
        void BFS(vector<int>& visitedNodes, vector<int>& distances, int startIndex = 0);
        void TarjanDFS(int currentNode, vector<vector<int>>& ssc, vector<int>& order, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& onStack, int& counter);
        void BiconnectedDFS(int currentNode, int parent, int currentDepth, vector< vector <int>>& bcc, vector<int>& depth, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& visited);

    public:

        int NumberOfComponents();
        vector<int> UnweightedDistances(int startIndex = 0);
        vector<vector <int>> StronglyConnectedComponents();
        vector<vector <int>> BiconnectedComponents();


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
                nodes.push_back(Node(0,i));
            }

            for(int i = 0; i < numberOfNodes; ++i) {
                vector<int> tempVector;
                adjancencyList.push_back(tempVector);
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

void Graph::BFS(vector<int>& visitedNodes, vector<int>& distances, int startIndex /*= 0*/) {
                                                                        // Breadth-first search that sets visited nodes and distance to each node from node with index startIndex
    queue<int> nodeQueue;           // Queue of nodes that haven't had their edges processed yet

    nodeQueue.push(startIndex);
    distances[startIndex] = 0;      // Distance to starting node

    while(!nodeQueue.empty()) {

        int topNode = nodeQueue.front();

        visitedNodes[topNode] = 1;

        for(int neighbor : adjancencyList[topNode]) {
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

    for(int neighborIndex : adjancencyList[nodeIndex]) {

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


    for(int neighbor: adjancencyList[currentNode]) {

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

void Graph::BiconnectedDFS(int currentNode, int parent, int currentDepth, vector< vector <int>>& bcc, vector<int>& depth, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& visited) {

    depth[currentNode] =  currentDepth;
    lowest[currentNode] = currentDepth;
    visited[currentNode] = true;
    nodeStack.push(currentNode);

    for(int child: adjancencyList[currentNode]) {

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
        --node1; --node2;                       // Necesar deoarece nodurile sunt indexate de la 1 pe infoarena/biconex

        adjancencyList[node1].push_back(node2);

        if(!directed) {

            adjancencyList[node2].push_back(node1);
        }
    }
    
}


int main() {

    ifstream inputFile("biconex.in"); 
    ofstream outputFile("biconex.out");

    int numberOfNodes, numberOfEdges;

    inputFile >> numberOfNodes >> numberOfEdges;

    Graph graph(numberOfNodes, numberOfEdges);

    graph.BuildFromAdjacencyList(inputFile);

    vector< vector< int > > bcc = graph.BiconnectedComponents();     

    int bccNumber = bcc.size();
    
    outputFile << bccNumber << '\n';

    for(int i = 0; i < bccNumber; ++i) {
        for(int node : bcc[i]) {
            
            outputFile << node +1 << ' ';           // +1 deoarece nodurile sunt indexate de la 1 pe infoarena
        }

        outputFile << '\n';
    } 
    return 0;
}