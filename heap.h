#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>

using std::vector;
using std::string;
using std::pair;

template <typename T>
class Heap {  

public:
 
    T Root();
    T Extract();
    void Pop();
    void Insert(T value);
    int Remove(T value);
    void Build(vector<T> values);
    vector<T> ExtractAll();

    bool Empty() {
        return tree.empty();
    }
    void Clear() {
        tree.clear();
    }

    Heap(bool maxHeap = false) {
        this->maxHeap = maxHeap;
    }

    Heap(vector<T> values, bool maxheap = false) {

        for(auto value : values) {
            Insert(value);
        }
    }

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
    
    if( node < tree.size() ) SwapNodes(node, tree.size()-1);
    tree.pop_back();
    Descend(0);
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
