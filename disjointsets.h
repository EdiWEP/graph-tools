#pragma once
#include<iostream>
#include<vector>
#include<algorithm>

using std::vector;


struct DisjointSets {    // Struct used as helper for creating and manipulating disjoint sets of nodes

    int Root(int node);
    void Union(int node1, int node2);

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
