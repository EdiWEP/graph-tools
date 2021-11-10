#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>

using std::vector;
using std::string;
using std::pair;

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
