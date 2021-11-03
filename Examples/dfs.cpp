// infoarena.com/problema/dfs
#include<iostream>
#include<fstream>
#include<vector>

using namespace std;


class Node {

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
        void BFS();

    public:

        int NumberOfComponents();
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

void Graph::BuildFromAdjacencyList(istream& inputStream) {           // Sets edges between nodes by reading adjancency list pairs from inputStream

    int node1, node2;

    for(int i = 0; i < numberOfEdges; ++i) {

        inputStream >> node1 >> node2;
        --node1; --node2; // pentru faptul ca se indexeaza de la 1 nodurile din infoarena/dfs

        adjancencyList[node1].push_back(node2);
        adjancencyList[node2].push_back(node1);
        
    }
    
}

void Graph::DFS(vector<int>& visitedNodes, int marker /*= 1*/, int nodeIndex /*= 0*/) {

    visitedNodes[nodeIndex] = marker;

    for(int neighborIndex : adjancencyList[nodeIndex]) {

        if(!visitedNodes[neighborIndex]) {
            DFS(visitedNodes, marker, neighborIndex);
        }
    }

}

int Graph::NumberOfComponents() {

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

int main() {

    ifstream inputFile("dfs.in"); 
    ofstream outputFile("dfs.out");

    int numberOfNodes, numberOfEdges;

    inputFile >> numberOfNodes >> numberOfEdges;

    Graph graph(numberOfNodes, numberOfEdges);

    graph.BuildFromAdjacencyList(inputFile);

    int numberOfComponents = graph.NumberOfComponents();

    outputFile << numberOfComponents;
 
    return 0;
}