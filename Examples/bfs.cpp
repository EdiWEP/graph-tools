// infoarena.com/problema/bfs
#include<iostream>
#include<fstream>
#include<vector>
#include<queue>

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

    public:

        int NumberOfComponents();
        vector<int> UnweightedDistances(int startIndex = 0);

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
    queue<int> nodeQueue;

    nodeQueue.push(startIndex);
    distances[startIndex] = 0;

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
                                        // Computes number of components in graph.
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

void Graph::BuildFromAdjacencyList(istream& inputStream) {           // Sets edges between nodes by reading adjancency list pairs from inputStream

    int node1, node2;

    for(int i = 0; i < numberOfEdges; ++i) {

        inputStream >> node1 >> node2;
        --node1; --node2;                                           // Deoarece nodurile sunt indexate de la 1 pe infoarena/bfs
        
        adjancencyList[node1].push_back(node2);

        if(!directed) {

            adjancencyList[node2].push_back(node1);
        }
    }
    
}


int main() {

    ifstream inputFile("bfs.in"); 
    ofstream outputFile("bfs.out");

    int numberOfNodes, numberOfEdges, startIndex;

    inputFile >> numberOfNodes >> numberOfEdges >> startIndex;

    --startIndex;

    Graph graph(numberOfNodes, numberOfEdges, true);

    graph.BuildFromAdjacencyList(inputFile);

    vector<int> distances = graph.UnweightedDistances(startIndex);

    for(int distance: distances) {
        outputFile << distance << ' ';
    }
    
    return 0;
}