#ifndef STOKE_SRC_SEARCH_MCTS_NODE_H
#define STOKE_SRC_SEARCH_MCTS_NODE_H

#include <vector>

#include "src/search/search_state.h"

namespace stoke {

struct Node {
  
  Node(Node* parent, SearchState& state) : 
    num_visit_(0),
    score_(0),
    state(state),
    parent(parent) {}
  int num_visit_;
  float score_;

  SearchState state;

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