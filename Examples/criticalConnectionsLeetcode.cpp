#include <algorithm>
#include <vector>
// Am folosit metoda CriticalEdgesDFS din clasa graphtools.h, adaptata la felul in care primeam inputul pe leetcode(in loc de vector <pair <int,int> > aici este vector<vector<int>>)
// Problema a avut runtime 616ms cu 193.5MB memorie folosita
class Solution {
public:
  void CriticalEdgesDFS(int currentNode, int parent, vector < vector <int>>& cc, vector<int>& order, vector<int>& lowest, int& counter, vector<vector<int>>& adjacencyList) {

    order[currentNode] = counter;
    lowest[currentNode] = counter;

    ++counter;

    for(int child : adjacencyList[currentNode]) {

        if(order[child] < 0) {

            CriticalEdgesDFS(child, currentNode, cc, order, lowest, counter, adjacencyList);

            lowest[currentNode] = min(lowest[currentNode], lowest[child]);      // Update lowest after subtree is finished

            if (lowest[child] > order[currentNode]) {
                                                            // We can't reach the current node through any path that doesn't include the current edge
                                                            // So current edge is critical
                vector<int> edge;
                edge.push_back( currentNode); edge.push_back(child);

                cc.push_back(edge);            
            }
        }
        else if (child != parent) {

            lowest[currentNode] = min(lowest[currentNode], order[child]);
        }
    }
}
    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
        
      vector< vector<int> > adjacencyList;
      
      for(int i = 0; i < n; ++i) {
        
        vector<int> tempVector;
        adjacencyList.push_back(tempVector);
        
      }
      
      for(int i = 0; i < connections.size(); ++i) {
        
        adjacencyList[connections[i][0]].push_back(connections[i][1]);
        adjacencyList[connections[i][1]].push_back(connections[i][0]);
        
      }
      
      vector<vector <int>> cc;                         // The list of critical edges

    vector<int> order;                  // Node X is the order[x]-th node to be found during DFS
    vector<int> lowest;                 // lowest[X] is the minimum order[Y], where node Y is connected to node X
     
    int counter = 0;

    for(int i = 0; i < n; ++i) {

        order.push_back(-1);            // -1 order means node hasn't been visited yet
        lowest.push_back(-1);
    }
      
      CriticalEdgesDFS(0, -1, cc, order, lowest, counter, adjacencyList);  // Root node has no parent

    return cc;
      
    }
};