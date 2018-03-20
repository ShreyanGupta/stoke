#ifndef STOKE_SRC_SEARCH_MCTS_NODE_H
#define STOKE_SRC_SEARCH_MCTS_NODE_H

#include <vector>

#include "src/cfg/cfg.h"
#include "src/cost/cost.h"

namespace stoke {

struct Node {
  
  Node(Node* parent) : num_visit_(0), score_(0), parent(parent) {}
  int num_visit_;
  float score_;

  Cfg cfg;
  Cost cost;

  Node* parent;
  std::vector<Node*> children;
  
  // Function to update score
  void update(float score){
    ++num_visit_;
    score_ += score;
  }
};

} // namespace stoke

#endif