#ifndef STOKE_SRC_SEARCH_MCTS_STATISTICS_H
#define STOKE_SRC_SEARCH_MCTS_STATISTICS_H

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

#include "src/search/mcts_node.h"

using namespace std;

namespace stoke {

struct MctsStatistics {
  
  MctsStatistics(size_t& num_mcts_itr_, size_t& num_mcmc_itr_, chrono::duration<double>& time_elapsed_) :
    num_mcts_itr_(num_mcts_itr_),
    num_mcmc_itr_(num_mcmc_itr_),
    num_nodes_(1),
    time_elapsed_(time_elapsed_) {}

  void print(Node* node) {
    cout << "***************************************" << endl;
    cout << "MCTS statistics" << endl;
    cout << "num_mcmc_itr : " << num_mcmc_itr_ << endl;
    cout << "num_mcts_itr : " << num_mcts_itr_ << endl;
    cout << "time_elapsed : " << time_elapsed_.count() << endl;
    cout << "mcmc itr per sec : " << num_mcmc_itr_/time_elapsed_.count() << endl;
    cout << "mcts itr per sec : " << num_mcts_itr_/time_elapsed_.count() << endl << endl;

    cout << "Best costs" << endl;
    cout << "time : cost" << endl;
    for(auto& i : time_cost_vec_){
      cout << " T : " << i.first << "\t C : " << i.second << endl;
    }
    cout << endl;

    int depth = -1;
    while(node != nullptr){
      ++depth;
      printf("%3d : (%.2f, %d) :\t", depth, (float)(node->score_/node->num_visit_), node->num_visit_);
      // vector of -%, score
      vector<pair<float, float>> v;
      for(Node* c : node->children){
        v.push_back(make_pair(-100.0*c->num_visit_/node->num_visit_, (float)(c->score_/c->num_visit_)));
        // printf("(%.2f, %.2f%%)\t", (float)(c->score_/c->num_visit_), 100.0*c->num_visit_/node->num_visit_);
      }
      // Sort by descending order of %
      sort(v.begin(), v.end());
      for(auto& i: v){
        printf("(%.2f, %.2f%%)\t", i.second, -i.first);
      }
      cout << endl;
      node = node->parent;
    }

    cout << "***************************************" << endl << endl;
  }
  
  size_t& num_mcts_itr_;
  size_t& num_mcmc_itr_;
  size_t num_nodes_;
  chrono::duration<double>& time_elapsed_;
  vector<pair<double, double>> time_cost_vec_;
};

} // namespace stoke

#endif
