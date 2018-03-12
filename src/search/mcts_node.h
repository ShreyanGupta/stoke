#ifndef STOKE_SRC_SEARCH_MCTS_NODE_H
#define STOKE_SRC_SEARCH_MCTS_NODE_H

#include <vector>

#include "src/transform/info.h"

namespace stoke {

struct Node {
  
  Node(Node* parent) : num_visit_(0), score_(0), parent(parent) {}
  int num_visit_;
  float score_;
  Node* parent;
  std::vector<Node*> children;
  std::vector<TransformInfo> ti_vector;
  
  // Function to update score
  void update(float score);
};

} // namespace stoke

#endif