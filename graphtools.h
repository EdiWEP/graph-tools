#include<iostream>
#include<fstream>
#include<vector>
#include<queue>
#include<stack>
#include<algorithm>

using namespace std;


// Auxiliary struct for holding edge information
struct Edge {                                       
        int source, destination, cost;          
        Edge(int source, int dest, int cost = 0): source(source), destination(dest), cost(cost){}
        // Returns edge with source and destination swapped
        Edge Flip() {return Edge(destination, source, cost);}   
};

// Class used for binary heap operations
template <typename T>
class Heap {  

public:
    
    // Returns maximum/minimum element
    T Root();
    // Returns maximum/minimum element and removes it from the heap
    T Extract();
    // Removes minimum/maximum element from the heap
    void Pop();
    // Inserts a element into the heap
    void Insert(T value);
    // Finds and removes an element by value from the heap
    // Does nothing if the element is not found
    int Remove(T value);
    // Inserts all elements of values into the heap
    void Build(vector<T> values);
    // Returns a vector of all elements in the heap and empties it
    vector<T> ExtractAll();

    // Returns true if heap is empty
    bool Empty() {
        return tree.empty();    
    }
    // Removes all elements from the heap
    void Clear() {
        tree.clear();
    }

    Heap(bool maxHeap = false) {
        this->maxHeap = maxHeap;
    }

    // Constructs a heap and inserts all elements of values into the heap
    Heap(vector<T> values, bool maxheap = false) {

        for(auto value : values) {
            Insert(value);
        }
    }

    // Returns the number of nodes currently in the heap
    int GetNumberOfNodes() {
        return tree.size();
    }

private:

    bool maxHeap;
    vector<T> tree;

    int Parent(int node);
    int LeftChild(int node);
    int RightChild(int node);
    void Descend(int node);
    void Ascend(int node);
    void Delete(int node);
    void SwapNodes(int node1, int node2);
    
    
    template<typename T0> static T0 Opposite(T0 x) {
        return -x;
    };
    static string Opposite(string x) {
        string newString = x;
        for(int i = 0; i < x.size(); ++i) {
            newString[i] = Opposite<char>(x[i]);
        }
        return newString;
    };
    template<typename T1, class T2> static pair<T1, T2> Opposite(pair<T1, T2> x)  {
        return make_pair(Opposite(x.first), x.second);
    };

};

#pragma region HeapPublicMethods

template<typename T>
T Heap<T>::Root() {  // Returns minimum/maximum element 

    if(maxHeap) {
        return Opposite(tree[0]);
    }
    else {
        return tree[0];
    }
}

template<typename T>
void Heap<T>::Insert(T value) { // Inserts element into heap

    T newValue = value;
    if(maxHeap) newValue = Opposite(value);

    tree.push_back(newValue);
    Ascend(tree.size() - 1);
}

template<typename T>
void Heap<T>::Pop() {  // Removes minimum/maximum element from the heap
    Delete(0);
}

template<typename T>
T Heap<T>::Extract() {  // Removes minimum/maximum element from the heap and returns it
    T root = Root();
    Delete(0);
    return root;
}

template<typename T>
int Heap<T>::Remove(T value) {   // Removes a given element from the heap, returns 0 if element was not found, returns 1 on succesful removal

    bool found = false;

    for(int i = 0; i < tree.size(); ++i) {
        if(tree[i] == value) {
            found = true;
            Delete(i);
            break;
        }
    }

    return found? 1 : 0;
}

template<typename T>
void Heap<T>::Build(vector<T> values) {   // Inserts all elements from values into the heap
    for(auto value : values) {
        Insert(value);
    }
}

template<typename T>
vector<T> Heap<T>::ExtractAll() {      // Returns a sorted vector of all elements

    vector<T> sortedVector;

    while(!Empty()) {
        sortedVector.push_back(Extract());
    }

    return sortedVector;
}

#pragma endregion

#pragma region HeapPrivateMethods

template<typename T>
void Heap<T>::Descend(int node) {

    T minimum = tree[node];
    int left = LeftChild(node);
    int right = RightChild(node);
    int minIndex = node;

    if(left) {
        if(minimum > tree[left]) {
            minIndex = left;
            minimum = tree[left];
        }
    }

    if(right) {
        if(minimum > tree[right]) {
            minIndex = right;
        }
    }

    if(node != minIndex) {
        SwapNodes(node, minIndex);
        Descend(minIndex);
    }
}    

template<typename T>
void Heap<T>::Ascend(int node) {

    int parent = Parent(node);

    if(tree[parent] > tree[node]) {
        SwapNodes(node, parent);
        Ascend(parent);
    }
}

template<typename T>
void Heap<T>::Delete(int node) {
    
    SwapNodes(node, tree.size()-1);
    tree.pop_back();
    Descend(node);
}

template<typename T>
void Heap<T>::SwapNodes(int node1, int node2) {

    T swap = tree[node1];
    tree[node1] = tree[node2];
    tree[node2] = swap;
}

template<typename T>
int Heap<T>::Parent(int node) {

    if(node > 0) {
        return (node - 1) / 2;
    }
    else return 0;
}

template<typename T>
int Heap<T>::LeftChild(int node) {
    int left = node * 2 + 1;

    if( left < tree.size()) {
        return left;
    }
    else return 0;
}

template<typename T>
int Heap<T>::RightChild(int node) {
    int right = node * 2 + 2;

    if( right < tree.size()) {
        return right;
    }
    else return 0;
}

#pragma endregion

// Struct used as helper for creating and manipulating disjoint sets of nodes
struct DisjointSets {    

    // Returns the root of the node's component
    int Root(int node);
    // Merges the components of node1 and node2
    // The smaller component's root is always attached to the larger one's root
    void Union(int node1, int node2);

    // Constructs and initializes numberOfNodes nodes into numberOfNodes disjoint sets of rank 1
    DisjointSets(int numberOfNodes): numberOfNodes(numberOfNodes) {

        for(int node = 0; node < numberOfNodes; ++node) {
            rank.push_back(1);
            parent.push_back(node);
        }
    }

    private:

        int numberOfNodes;

        vector<int> rank;           // Holds the rank of the set of each node, for optimizing unions
        vector<int> parent;         // Either the direct parent's index or the root node index 

};

#pragma region DisjointSetsPublicMethods

int DisjointSets::Root(int node) {        // Returns the root of the node's subtree

    int root = node;

    while(parent[root] != root) {
        root = parent[root];        // Go up through the set until its root is found
    }


    while(node != parent[node]) {   // Update parent[node] to the set's root, for all nodes up to the root
        int swap = parent[node];
        parent[node] = root;
        node = swap; 
    }

    return root;
}

void DisjointSets::Union(int node1, int node2) {  // Merges the two subtrees by connecting one's root to the other's
    node1 = Root(node1);            // Get the roots 
    node2 = Root(node2);

    if (rank[node1] > rank[node2]) {    // Link roots, smaller rank gets attached to higher rank 
        parent[node2] = node1;
    }
    else {
        parent[node1] = node2;
    }
    if(rank[node1] == rank[node2]) {
        ++rank[node2];              // Increase rank of new set if sets were equal 
    }
}

#pragma endregion

class FlowHandler;

class Graph {

public:

        // Holds the max value for int, used to represent infinite distances(for unreachable nodes)
        // = 2147483647 
        static const int INFINITE; 
        // Returns the sum of all edge weights
        int TotalCost();                                        
        // Returns the number of connected components (for undirected graphs)
        int NumberOfComponents();        
        // Returns the diameter of the graph(length of longest sequence of connected nodes). Should only be used on trees
        int TreeDiameter();              
        // Returns the maximum flow from source to sink. Should only be used after BuildFlowNetwork() has been called
        // Requires handler to be the FlowHandler that was passed to BuildFlowNetwork()
        int MaximumFlow(unsigned int source, unsigned int sink, FlowHandler& handler);         
        // Returns all edges in the graph
        vector<Edge> GetAllEdges();                             
        // Returns a vector of distances from node of index startIndex to all others, ignoring weights
        vector<int> UnweightedDistances(int startIndex = 0);    
        // Returns a vector of distances from node of index startIndex to all others 
        // Uses Dijkstra's algorithm if there are no negative edges, otherwise uses the Bellman-Ford algorithm
        vector<int> WeightedDistances(int startIndex = 0);      
        // Returns an adjacency matrix, returnMatrix[i][j] represents the cost of the minimum path from node i to node j
        // returnMatrix[i][j] == Graph::INFINITE means there is no edge from i to j
        vector<vector <int>> AllMinimumDistances();
        // Returns a vector containing the topological sort of current graph (for directed graphs)
        vector<int> TopologicalSort();                          
        // Returns vectors of node indexes, grouped into the graph's strongly connected components (for directed graphs)
        vector<vector <int>> StronglyConnectedComponents();     
        // Returns vectors of node indexes, grouped into the graph's biconnected components 
        vector<vector <int>> BiconnectedComponents();  
        // Returns a vector of the critical edges in the graph
        vector<Edge> CriticalConnections();                     

        // Returns a new graph that is the MST of this graph
        Graph MinimumSpanningTree();
        // Returns a new graph that is the DF search tree of this graph starting in startIndex
        Graph DFSTree(int startIndex);
        // Returns a new graph that is the BF search tree of this graph starting in startIndex
        Graph BFSTree(int startIndex);
        // Returns a new graph that contains all DFS trees of this graph
        // Starts a DFS from index 0, then visits all unvisited nodes, adding the other trees to the returned graph
        Graph DFSTrees();
        
        // Reads edges from inputStream, formated as adjacency list entries: nodeX, nodeY, capacity, cost
        // and builds a flow network graph using handler as a helper for operations on the network
        // Should only be used after setting numberOfNodes and numberOfEdges 
        void BuildFlowNetwork(istream& inputStream, FlowHandler& handler);

        // Reads edges from inputStream, formated as adjacency list entries: nodeX, nodeY, cost
        // Should only be used if numberOfEdges has already been set
        void BuildFromAdjacencyList(istream& inputStream);
        
        // Reads edges from inputStream, formated as an adjacency matrix: 
        // 0 represents no edge, any other value represents the cost of the edge
        // If upon calling numberOfEdges == 0 (has not been set), sets numberOfEdges according to the matrix
        void BuildFromAdjacencyMatrix(istream& inputStream);

        // Returns an adjacency matrix for the graph, returnMatrix[i][j] represents the cost from node i to node j 
        // returnMatrix[i][j] == Graph::INFINITE means there is no edge from i to j
        vector<vector<int>> GetAdjacencyMatrix();
        // Adds a new edge to the graph, increases numberOfEdges
        void AddEdge(int source, int dest, int cost = 0);
        // Adds a new edge to the graph, increases numberOfEdges
        void AddEdge(Edge newEdge);
        // Adds a new node to the graph, increases numberOfNodes
        void AddNode(int value = 0);

        // Takes a vector of node degree values and checks if a graph can be formed using the Havel-Hakimi algorithm 
        static bool CheckHavelHakimi(vector<int> degrees);

        

#pragma region GraphConstructors

        Graph(bool directed = false, bool weighted = false) {

            numberOfNodes = 0;
            numberOfEdges = 0;
            this->directed = directed;
            this->weighted = weighted; 
        }

        Graph(int numberOfNodes, int numberOfEdges = 0, bool directed = false, bool weighted = false) {
            
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

        int GetValue(int node) const {
            
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

        bool IsDirected() const {
            return directed;
        }

        void SetWeighted(bool weighted) {
            this->weighted = weighted;
        }

        bool IsWeighted() const {
            return weighted;
        }

        void SetNumberOfNodes(int number) {
            numberOfNodes = number;
        }
        
        int GetNumberOfNodes() const {
            return numberOfNodes;
        }

        void SetNumberOfEdges(int number) {
            numberOfEdges = number;
        }

        int GetNumberOfEdges() const {
            return numberOfEdges;
        }

#pragma endregion

    private:

        bool directed;
        bool weighted;

        int numberOfNodes;
        int numberOfEdges;

        vector < int > values;                              // The value of each node 
        vector < vector < Edge > > adjacencyList;           // adjacencyList[X] holds the information of each edge that X has

        
        void Dijkstra(vector<bool>& visitedNodes, vector<int>& distances, Heap< pair<int, int>>& heap, int startIndex = 0);        
        void BellmanFord(vector<bool>& betterPath, vector<int>& distances, Heap<pair<int, int>>& heap, int startIndex = 0);
        void BFS(vector<bool>& visitedNodes, vector<int>& distances, int startIndex = 0);
        void FlowBFS(vector<bool>& visitedNodes, vector<int>& prev, FlowHandler& fh, int startIndex, int sink);
        void TreeBuilderBFS(int startIndex, Graph& treeGraph, vector<bool>& visitedNodes);
        void DFS(vector<int>& visitedNodes, int marker = 1, int nodeIndex = 0);
        void StronglyConnectedDFS(int currentNode, vector<vector<int>>& scc, vector<int>& order, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& onStack, int& counter);
        void BiconnectedDFS(int currentNode, int parent, int currentDepth, vector< vector <int>>& bcc, vector<int>& depth, vector<int>& lowest, stack<int>& nodeStack, vector<bool>& visited);
        void CriticalEdgesDFS(int currentNode, int parent, vector <Edge>& cc, vector<int>& depth, vector<int>& lowest, int& counter);
        void TopologicalDFS(int currentNode, stack<int>& orderStack, vector<bool>& visitedNodes);
        void TreeBuilderDFS(int currentNode, int treeCurrentNode, Graph& treeGraph, vector<bool>& visitedNodes);

        bool HasNegativeEdges();
        static bool CompareEdges(Edge x, Edge y);
    
};
const int Graph::INFINITE = 2147483647;

// Class used as a helper for graph class, to be passed as parameter to graph methods that operate on flow networks 
// Should be instantianted after initializing graph with numberOfNodes and numberOfEdges
class FlowHandler {

    friend class Graph;

public:

    // The matrix for storing capacity of each edge. capacityMatrix[x][y] holds the capacity of the edge x -> y
    vector< vector< int >> capacityMatrix;

    // Must receive a pointer to the graph that represents the flow network and the indexes of the source and sink nodes
    FlowHandler(Graph* graphPointer, int sourceIndex, int sinkIndex): graph(graphPointer), source(sourceIndex), sink(sinkIndex) {
        
        numberOfNodes = graph->GetNumberOfNodes();

        for(int i = 0; i < numberOfNodes; ++i) {
        
            vector<int> tempVector;
            capacityMatrix.push_back(tempVector);
            flowMatrix.push_back(tempVector);
            for(int j = 0; j < numberOfNodes; ++j) {
                capacityMatrix[i].push_back(0);
                flowMatrix[i].push_back(0);
            }
        }
    }
    
    int GetSource() {return source;}
    int GetSink() {return sink;}

    void SetSource(unsigned int source) { 
        try {
            if (source < numberOfNodes) this->source = source;
            else throw;
        }
        catch (...) {
            throw "Source index out of range";
        }
    }

    void SetSink(unsigned int sink) {
        try {
            if (sink < numberOfNodes) this->sink = sink;
            else throw;
        }
        catch (...) {
            throw "Sink index out of range";
        }
    }

private:
    // The graph this FlowHandler is corelated to
    const Graph* graph; 
    int numberOfNodes;
    int source, sink;

    vector< vector< int >> flowMatrix;
};

#pragma region GraphPublicMethods

void Graph::AddEdge(int source, int dest, int cost /*= 0*/) {

    Edge newEdge(source, dest, cost);

    adjacencyList[source].push_back(newEdge);

    if(!this->directed) {

        adjacencyList[dest].push_back(newEdge.Flip());
    }

    ++numberOfEdges;
}

void Graph::AddEdge(Edge newEdge) {

    adjacencyList[newEdge.source].push_back(newEdge);

    if(!this->directed) {

        adjacencyList[newEdge.destination].push_back(newEdge.Flip());
    }

    ++numberOfEdges;
}

void Graph::AddNode(int value /*= 0*/) {

    vector<Edge> tempVector;
    adjacencyList.push_back(tempVector);

    values.push_back(value);

    ++numberOfNodes;
}

vector<Edge> Graph::GetAllEdges() {

    vector<Edge> edges;

    if (directed) {
        for(int node = 0; node < numberOfNodes; ++node) {
            for(auto edge : adjacencyList[node]) {
                edges.push_back(edge);            
            }
        }
    }
    else {
    
        vector<bool> visited;                   // Used for not returning duplicate edges in undirected graphs

        for(int i = 0; i < numberOfNodes; ++i) {
            visited.push_back(false);
        }

        for(int node = 0; node < numberOfNodes; ++node) {
            for(auto edge : adjacencyList[node]) {
                if (!visited[edge.destination]) {
                    edges.push_back(edge);
                }
            }
        
        visited[node] = true;

        }
    }

    return edges;
}

Graph Graph::MinimumSpanningTree() {
                                                // Returns a graph that is a minimum spanning tree of this graph, using Kruskal's algorithm and disjoint sets

    Graph treeGraph(numberOfNodes, 0, directed, weighted);  // Initialize graph to be returned
                                                            // Has 0 edges currently because numberOfEdges will be incremented by AddEdge()
    vector<Edge> sortedEdges = GetAllEdges();
    sort(sortedEdges.begin(), sortedEdges.end(), CompareEdges);

    DisjointSets sets(numberOfNodes);           // Used for keeping track of the new trees components

    for(int i = 0; i < numberOfEdges; ++i) {

        if( sets.Root(sortedEdges[i].source) != sets.Root(sortedEdges[i].destination) )  {       // Check if nodes are in separate components

            sets.Union(sortedEdges[i].source, sortedEdges[i].destination);
            treeGraph.AddEdge(sortedEdges[i]);
        }
    }

    return treeGraph;
}

int Graph::TotalCost() {
                            // Returns the summed cost of all edges in graph
    int totalCost = 0;

    vector<Edge> edges = GetAllEdges();

    for(auto edge : edges) {
        
        totalCost += edge.cost;
    }

    return totalCost;
}

int Graph::TreeDiameter() {

    vector<int> distances = UnweightedDistances(); // Perform a BFS

    int maximum = 0;
    int maxIndex;

    for(int i = 0; i < numberOfNodes; ++i) {
        if (distances[i] > maximum) {
            maximum = distances[i];
            maxIndex = i;
        }
    }

    distances = UnweightedDistances(maxIndex);  // Perform a second BFS from the node that's farthest away
    maximum = 0;

    for(int i = 0; i < numberOfNodes; ++i) {
        if(distances[i] > maximum) {
            maximum = distances[i];
        }
    }

    return maximum + 1; // Add 1 to count starting node 
}

int Graph::MaximumFlow(unsigned int source, unsigned int sink, FlowHandler& handler) {

    int totalFlow = 0;

    vector<bool> visited; // For marking visited nodes
    vector<int> previous; // Used for tracking paths to sink 

    for(int i = 0; i < numberOfNodes; ++i) {    // Initialization
        visited.push_back(false);
        previous.push_back(0);
    }
    
    FlowBFS(visited, previous, handler, source, sink);

    while(visited[sink]) {

        for(auto edge : adjacencyList[sink]) {  // By checking the paths of all nodes that connect to the sink, we can reduce the number of BFS's that must be performed

            previous[sink] = edge.destination;
            int minimumIncrease = Graph::INFINITE;
            int node = sink; // Current node in path
            
            while(node != source) { // Go back through the path to find the minimum possible increase for the flow of the whole path
                int prev = previous[node];
                int increase = handler.capacityMatrix[prev][node] - handler.flowMatrix[prev][node];

                if(minimumIncrease > increase) {
                    minimumIncrease = increase;
                }
                
                node = prev;
            }

            
            if(minimumIncrease != 0) { // If an improvement was found 
                node = sink;
                while(node != source) { // Go back through the path to find the minimum possible increase for the flow of the whole path
                    int prev = previous[node];
                    
                    handler.flowMatrix[prev][node] += minimumIncrease;
                    handler.flowMatrix[node][prev] -= minimumIncrease; // Set residual graph edge 

                    node = prev;
                }

                totalFlow += minimumIncrease;
            }

        }

        for(int i = 0; i < numberOfNodes; ++i) {
            visited[i] = false;
        }
        FlowBFS(visited, previous, handler, source, sink);
    }

    return totalFlow;
}

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

bool Graph::CheckHavelHakimi(vector<int> degrees) {
                                                                            // Receives a list of node degrees and returns true if a corelated graph can exist through the Havel-Hakimi algorithm
   
    int maximum = -1; 
    int sum = 0;  
    int numberOfNodes = degrees.size();

    for(int degree : degrees) {

        if(maximum < degree) {
            maximum = degree;
        }
        sum += degree;
        if (degree >= numberOfNodes) {
                                    // If degree is > n-1 then there are not enough nodes, therefore a graph doesn't exist
            return false; 
        }
    }

    if(sum % 2) {                           // If sum of degrees is odd then a graph doesn't exist
        return false; 
    }

    vector<int> frequency(maximum + 1, 0);
    
    for(int degree : degrees) {
        ++frequency[degree];    // Form frequency vector
    }

    stack<pair<int, int>> changeStack;  // Stack used to keep track of changes on the frequency vector
                                        // .first is the node whose frequency number has to have .second added to it
    while(sum > 0) {
        pair <int, int> stackTop;
        
        while(!changeStack.empty()) {
            stackTop = changeStack.top();
            frequency[stackTop.first] += stackTop.second;
            changeStack.pop();

            if(stackTop.first > maximum) maximum = stackTop.first;
        }

        while(frequency[maximum] == 0) {    // Find first value still in vector
            --maximum;
            if(maximum < 0) return false;       // There are no more numbers, so there are too many subtractions, therefore no graph exists
        }

        int subtract = maximum; // Value to be subtracted from vector
        int oldMax = maximum;
        --frequency[maximum];
        
        while(subtract > 0) {

            while(frequency[maximum] == 0) {    // Find first value still in vector
                --maximum;
                if(maximum < 0) return false;       // There are no more numbers, so there are too many subtractions, therefore no graph exists
            }

            if(frequency[maximum] >= subtract) {
                if(maximum > 1) changeStack.push(make_pair(maximum - 1, subtract));
                frequency[maximum] -= subtract;
                subtract = 0;
            }
            else {
                if(maximum > 1) changeStack.push(make_pair(maximum - 1, frequency[maximum])); 
                subtract -= frequency[maximum];
                frequency[maximum] = 0;
            }

        }
        sum -= 2*oldMax;
    }
    return true;            // If this point is reached then the vector is now [0, 0, .. 0] and a graph exists for the given set of degrees          
}

vector<Edge> Graph::CriticalConnections() {
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
    
    vector<bool> visited;
    vector<int> distances;

    
    for(int i = 0; i < numberOfNodes; ++i) {
        visited.push_back(0);
        distances.push_back(-1);                // -1 means there is no path to a certain node
    }

    BFS(visited, distances, startIndex);

    return distances;
}

vector<int> Graph::WeightedDistances(int startIndex /*= 0*/) {
                                                // Computes the weighted distances from node of index startIndex to all other nodes
                                                // Uses BellmanFord instead of Dijkstra if there are negative weight edges
                                                // Return vector has distances[startIndex] == 1 if there are negative weight cycles
    vector<int> distances;
    vector<bool> visited;
    Heap< pair<int, int> > heap; 
    // The heap will contain pairs of (distance, node), where distance is the current total distance to node
        
    int infinity = 2147483647;      // Used for initializing distance vector

    for(int i = 0; i < numberOfNodes; ++i) {

        if(i == startIndex) {
            distances.push_back(0);     
        }
        else {
            distances.push_back(infinity);
        }

        visited.push_back(false);
    }

    if(HasNegativeEdges()) {
        BellmanFord(visited, distances, heap, startIndex);
    }   
    else { 
        Dijkstra(visited, distances, heap, startIndex);
    }
    return distances;
}

vector<vector <int>> Graph::AllMinimumDistances() {

    vector<vector<int>> adjMatrix = GetAdjacencyMatrix();

    for(int node = 0; node < numberOfNodes; ++node) {    // For each node, check if it can be used to make any path shorter
        for(int i = 0; i < numberOfNodes; ++i) {
            for(int j = 0; j < numberOfNodes; ++j) {
                
                if(i == j) continue;
                if(adjMatrix[i][node] != INFINITE && adjMatrix[node][j] != INFINITE ) {  // If a path from i to j through node exists

                    int detourPath = adjMatrix[i][node] + adjMatrix[node][j];
                    if(adjMatrix[i][j] > detourPath || adjMatrix[i][j] == INFINITE ) {
                        adjMatrix[i][j] = detourPath;
                    }
                }
            }
        }
    }

    return adjMatrix;

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

vector<vector<int>> Graph::GetAdjacencyMatrix() {

    vector<vector<int>> matrix;

    for(int i = 0; i < numberOfNodes; ++i) {
        vector<int> tempVector(numberOfNodes, INFINITE);
        matrix.push_back(tempVector);
    }

    for(int i = 0; i < numberOfNodes; ++i) {
        for(auto edge : adjacencyList[i]) {

            matrix[i][edge.destination] = edge.cost;
        }
    }

    return matrix;
}

void Graph::BuildFromAdjacencyMatrix(istream& inputStream) {        // Sets edges between nodes by reading an adjacency matrix from inputStream
                                                                    // Should only be used if numberOfEdges has already been set 

    bool incrementNumber = false;
    if(!numberOfEdges) {
        incrementNumber = true;
    }

    int matrixValue;             // If matrixValue is 0, there is no edge, if matrixValue != 0, then there is an edge of cost matrixValue

    for(int i = 0; i < numberOfNodes; ++i) {
        for (int j = 0; j < numberOfNodes; ++j) {

            inputStream >> matrixValue;

            if(matrixValue) {
                
                if(!weighted) {
                    matrixValue = 0;
                }

                Edge newEdge(i, j, matrixValue);

                if(incrementNumber) {
                    AddEdge(newEdge);
                    continue;
                }
                
                adjacencyList[i].push_back( newEdge );

                if(!directed) {    
                    adjacencyList[j].push_back( newEdge.Flip() );
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
                adjacencyList[node2].push_back( newEdge.Flip() );
        }
    }   
}

void Graph::BuildFlowNetwork(istream& inputStream, FlowHandler& handler) {

    int node1, node2, capacity, cost;

    for(int i = 0; i < numberOfEdges; ++i) {

        inputStream >> node1 >> node2 >> capacity;
        --node1; --node2; // deoarece pe infoarena nodurile sunt indexate de la 1

        if(weighted) {
            inputStream >> cost;
        }
        else {
            cost = 0;
        }
        
        Edge newEdge(node1, node2, cost);
        adjacencyList[node1].push_back( newEdge );
        adjacencyList[node2].push_back( newEdge.Flip() );

        handler.capacityMatrix[node1][node2] = capacity;

    }
}

#pragma endregion

#pragma region GraphPrivateMethods

bool Graph::HasNegativeEdges() {

    vector<Edge> edges = GetAllEdges();
    for(auto edge : edges) {
        if ( edge.cost < 0 ) return true;
    }
    return false;
}

bool Graph::CompareEdges(Edge x, Edge y) {
    return (x.cost < y.cost); 
}

void Graph::Dijkstra(vector<bool>& visitedNodes, vector<int>& distances, Heap < pair<int, int> >& heap, int startIndex /*= 0*/) {

    heap.Insert(make_pair(0, startIndex));    // Push starting node

    while(!heap.Empty()) {

        int currentDistance = heap.Root().first;
        int currentNode = heap.Root().second;

        visitedNodes[currentNode] = true;
        heap.Pop();

        if(distances[currentNode] < currentDistance) {  // If the current minimum distance is smaller than the one in the current pair, 
                                                        // then there is no point in processing the pair further, 
            continue;                                   // because it can't lead to shorter paths
        }

        for(auto edge : adjacencyList[currentNode]) {

            if(!visitedNodes[edge.destination]) {       // If we have already processed a node, then the minimum distance to it has already been calculated

                if(distances[edge.destination] > currentDistance + edge.cost) {
                    
                    distances[edge.destination] = currentDistance + edge.cost;
                    heap.Insert(make_pair(distances[edge.destination], edge.destination));
                }
            }
        }
    }
}   

void Graph::BellmanFord(vector<bool>& betterPath, vector<int>& distances, Heap<pair<int, int>>& heap, int startIndex /*= 0*/) {

    heap.Insert(make_pair(0, startIndex));

    long long loopCounter = 0;  // Used to count the number of loops in while cycle
                                // If loopCounter goes above (E-1)*V, 
                                // then there must be a negative weight cycle in the graph
    while(!heap.Empty()) {

        if(loopCounter > 1LL* numberOfEdges * (numberOfNodes - 1)) { // Too many improvements, there is a negative weight cycle
            distances[startIndex] = 1; // Marker for negative weight cycle
            break;
        }
        
        int currentNode = heap.Extract().second;
        betterPath[currentNode] = false;

        for(auto edge : adjacencyList[currentNode]) {

            if(distances[edge.destination] > distances[currentNode] + edge.cost) {  // Found a shorter path

                distances[edge.destination] = distances[currentNode] + edge.cost;

                if(!betterPath[edge.destination]) { // Found a new relevant node, so insert it into the heap
                    
                    betterPath[edge.destination] = true;
                    heap.Insert(make_pair(distances[edge.destination], edge.destination));
                }
            }
        }

        ++loopCounter;
    }
}

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

void Graph::BFS(vector<bool>& visitedNodes, vector<int>& distances, int startIndex /*= 0*/) {
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

void Graph::FlowBFS(vector<bool>& visitedNodes, vector<int>& prev, FlowHandler& fh, int source, int sink) {

    // Performs BF searches for Edmonds-Karp algorithm, setting the previous node for each node it reaches
    queue<int> nodeQueue;
    nodeQueue.push(source);

    while(!nodeQueue.empty()) {

        int currentNode = nodeQueue.front();
        nodeQueue.pop();

        visitedNodes[currentNode] = true;

        if(currentNode != sink) {

            for(auto edge: adjacencyList[currentNode]) {
                int neighbor = edge.destination;

                if(!visitedNodes[neighbor] && fh.flowMatrix[currentNode][neighbor] < fh.capacityMatrix[currentNode][neighbor]) { // If more flow can fit
                    
                    prev[neighbor] = currentNode;
                    nodeQueue.push(neighbor);
                }
            }
        }
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

