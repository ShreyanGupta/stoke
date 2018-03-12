#include <fstream>
#include <iostream>

#include <cassert>
#include <csignal>
#include <cmath>
#include <climits>

#include <map>
#include <utility>

#include "src/cost/cost.h"
#include "src/search/mcts.h"
#include "src/search/mcts_node.h"
#include "src/search/score_aggregator.h"
#include "src/transform/weighted.h"

using namespace cpputil;
using namespace std;
using namespace std::chrono;
using namespace x64asm;

namespace {

bool give_up_now = false;

void handler(int sig, siginfo_t* siginfo, void* context) {
  give_up_now = true;
}

void update_global_state(stoke::SearchState& curr_state, stoke::SearchState& state){
  if(curr_state.best_yet_cost <= state.best_yet_cost){
    state.best_yet = curr_state.best_yet;
    state.best_yet_cost = curr_state.best_yet_cost;
  }

  if(curr_state.success && (curr_state.best_correct_cost <= state.best_correct_cost)){
    state.success = true;
    state.best_correct = curr_state.best_correct;
    state.best_correct_cost = curr_state.best_correct_cost;
  }
}

} // namespace

namespace stoke {

void Node::update(float score){
  ++num_visit_;
  score_ += score;
}

Mcts::Mcts(Transform* transform) : 
  root_(new Node(nullptr)),
  transform_(transform), 
  num_mcmc_itr_(0),
  mcts_statistics_(MctsStatistics(num_itr_, time_elapsed_)) {

  set_seed(0);
  set_timeout_itr(0);
  set_timeout_sec(steady_clock::duration::zero());
  set_progress_callback(nullptr, nullptr);
  set_statistics_callback(nullptr, nullptr);
  set_statistics_interval(100000);
  set_mcts_args(1, 1000, 4, 1000, 1);

  static bool once = false;
  if (!once) {
    once = true;

    struct sigaction term_act;
    memset(&term_act, '\0', sizeof(term_act));
    sigfillset(&term_act.sa_mask);
    term_act.sa_sigaction = handler;
    term_act.sa_flags = SA_ONSTACK;

    sigaction(SIGINT, &term_act, 0);
  }
}

Node* Mcts::traverse(SearchState& state, int depth){
  Node* curr_node = root_;
  int curr_depth = 0;
  while(curr_node->children.size() != 0 && curr_depth != depth){
    int best_child_index = -1;
    // TODO : Find the limit of the highest score
    float least_score = 1000000;
    auto& children = curr_node->children;
    auto& ti_vector = curr_node->ti_vector;
    assert(children.size() == ti_vector.size());
    for(size_t i=0; i<children.size(); ++i){
      auto* child = children[i];
      auto& ti = ti_vector[i];
      float score = node_score(child);
      // Pick the lowest score
      if(score < least_score){
        least_score = score;
        best_child_index = i;
      }
    }
    assert(best_child_index != -1);
    curr_node = children[best_child_index];
    (*transform_).redo(state.current, ti_vector[best_child_index]);
    ++curr_depth;
  }
  return curr_node;
}

void Mcts::expand(Node* node, SearchState& state, CostFunction& fxn){
  // Stores the top k_ transformations.
  std::multimap<Cost, TransformInfo> m;
  for(int i=0; i<k_; ++i){
    // TODO : Find the limit of the highest score
    m.insert(std::make_pair((Cost)(100000), TransformInfo()));
  }

  // Search over a space of c_ transformations
  for(int i=0; i < std::max(c_,2*k_); ++i){
    TransformInfo ti = (*transform_)(state.current);
    if(!ti.success) continue;

    const auto max_cost = m.rbegin()->first;
    const auto new_res = fxn(state.current, max_cost + 1);
    const auto new_cost = new_res.second;

    if(new_cost < max_cost){
      m.insert(std::make_pair(new_cost, ti));
      m.erase(--m.end());
    }

    (*transform_).undo(state.current, ti);
    assert((int)m.size() == k_);
  }

  auto& children = node->children;
  auto& ti_vector = node->ti_vector;
  assert(children.size() == 0);

  for(auto& itr : m){
    if(!itr.second.success) continue;
    children.push_back(new Node(node));
    ti_vector.push_back(itr.second);
    mcts_statistics_.num_nodes_++;
  }
}

float Mcts::rollout(Node* node, SearchState& state, CostFunction& fxn){
  // Average out scored from all rollouts
  AvgAggregator score;
  // Rollout n times
  for(int i=0; i<n_; ++i){
    
    // Copy the init state
    SearchState curr_state = state;

    // Aggregator for specific rollout
    AvgAggregator agg;
    
    // Rollout for a depth of r
    for(int j=0; j<r_; ++j){
      // MCMC till depth r
      num_mcmc_itr_++;
      if ((statistics_cb_ != nullptr) && (num_mcmc_itr_ % statistics_interval_ == 0)) {
        statistics_cb_(get_statistics());
      }
      TransformInfo ti = (*transform_)(curr_state.current);
      move_statistics_[ti.move_type].num_proposed++;
      if(!ti.success) continue;

      move_statistics_[ti.move_type].num_succeeded++;

      const double p = prob_(gen_);
      const double max = curr_state.current_cost - (log(p) / beta_);

      const auto new_res = fxn(curr_state.current, max + 1);
      const bool is_correct = new_res.first;
      const auto new_cost = new_res.second;

      if(new_cost > max){
        (*transform_).undo(curr_state.current, ti);
        continue;
      }

      move_statistics_[ti.move_type].num_accepted++;

      curr_state.current_cost = new_cost;
      const bool new_best_yet = new_cost < curr_state.best_yet_cost;
      if (new_best_yet) {
        curr_state.best_yet = curr_state.current;
        curr_state.best_yet_cost = new_cost;
      }
      
      const bool new_best_correct_yet = is_correct && ((new_cost == 0) || (new_cost < curr_state.best_correct_cost));
      if (new_best_correct_yet) {
        curr_state.success = true;
        curr_state.best_correct = curr_state.current;
        curr_state.best_correct_cost = new_cost;
        if(new_best_correct_cb_ != nullptr){
          new_best_correct_cb_({curr_state});
        }
      }

      if ((progress_cb_ != nullptr) && (new_best_yet || new_best_correct_yet)) {
        progress_cb_({curr_state});
      }

      agg += curr_state.current_cost;
    }

    score += agg.get_score();

    // Update the global state
    update_global_state(curr_state, state);
  }
  return score.get_score();
}

void Mcts::update(Node* node, float score){
  // Do no need SearchState to update
  Node* curr_node = node;
  while(curr_node != nullptr){
    curr_node->update(score);
    curr_node = curr_node->parent;
  }
}

float Mcts::node_score(Node* node){
  assert(node->num_visit_ != 0);

  float x = node->score_ / node->num_visit_;
  float confidence = sqrt(2*log(num_itr_ + 1) / node->num_visit_);
  
  // TODO : Figure out the limits!!
  assert(x > -10000);
  assert(10000 > x);
  assert(confidence > -10000);
  assert(10000 > confidence);
  
  // -ve confidence as we are choosing the least value
  return x - exploration_factor_ * confidence;
}

void Mcts::trim(SearchState& state, int depth){
  Node* new_root = traverse(state, depth);
  delete_node(root_, new_root);
  root_ = new_root;
  root_->parent = nullptr;
}

void Mcts::delete_node(Node* node, Node* new_root){
  if(node == nullptr) return;
  if(node == new_root) return;
  for(auto* child : node->children){
    delete_node(child, new_root);
  }
  --mcts_statistics_.num_nodes_;
  delete node;
}

void Mcts::run(const Cfg& target, CostFunction& fxn, Init init, SearchState& state, vector<TUnit>& aux_fxns){

  // Configure initial state
  configure(target, fxn, state, aux_fxns);

  // Make sure target and rewrite are sound to begin with
  assert(state.best_yet.is_sound());
  assert(state.best_correct.is_sound());

  // Statistics callback variables. earch only works with 'WeightedTransform'
  move_statistics_ = vector<Statistics>(static_cast<WeightedTransform*>(transform_)->size());
  const auto start_time = chrono::steady_clock::now();

  // Early corner case bailouts
  if (state.current_cost == 0) {
    state.success = true;
    state.best_correct = state.current;
    state.best_correct_cost = 0;
    return;
  }

  give_up_now = false;

  for(num_itr_ = 0; true; ++num_itr_){
    // When to exit the loop
    time_elapsed_ = duration_cast<duration<double>>(steady_clock::now() - start_time);
    if(
      num_itr_ >= timeout_itr_ ||
      (timeout_sec_ != steady_clock::duration::zero() && time_elapsed_ >= timeout_sec_) ||
      state.current_cost <= 0 ||
      give_up_now
    ) break;

    // Invoke statistics callback if we've been running for long enough
    if(num_itr_ % mcts_statistics_interval_ == 0){
      // Use private copy of search_state
      SearchState curr_state = state;
      mcts_statistics_.print(traverse(curr_state));
    }

    // Work with current_state
    SearchState curr_state = state;

    // Actual loop
    Node* leaf = traverse(curr_state);
    expand(leaf, curr_state, fxn);
    for(auto* child : leaf->children){
      float score = rollout(child, curr_state, fxn);
      update(child, score);
    }

    // Update global state
    update_global_state(curr_state, state);

  } // End of loop

  if (give_up_now) {
    state.interrupted = true;
  }

  // make sure Cfg's are in a valid state (e.g. liveness information, which we
  // do not update during search)
  state.current.recompute();
  state.best_correct.recompute();
  state.best_yet.recompute();
}

StatisticsCallbackData Mcts::get_statistics() const {
  return {move_statistics_, num_mcmc_itr_, time_elapsed_, transform_};
}

void Mcts::stop() {
  give_up_now = true;
}

void Mcts::configure(const Cfg& target, CostFunction& fxn, SearchState& state, vector<TUnit>& aux_fxn) const {
  state.current.recompute();
  state.best_yet.recompute();
  state.best_correct.recompute();

  // add dataflow information about function call targets
  for (const auto& fxn : aux_fxn) {
    const auto& code = fxn.get_code();
    const auto& lbl = fxn.get_leading_label();
    TUnit::MayMustSets mms = {
      code.must_read_set(),
      code.must_write_set(),
      code.must_undef_set(),
      code.maybe_read_set(),
      code.maybe_write_set(),
      code.maybe_undef_set()
    };
    state.current.add_summary(lbl, fxn.get_may_must_sets(mms));
  }

  state.current_cost = fxn(state.current).second;
  state.best_yet_cost = fxn(state.best_yet).second;
  state.best_correct_cost = fxn(state.best_correct).second;
  state.success = false;

  // @todo -- Let's move these invariants into SearchState
  // Redirecting the user here to reason about this seems like an opportunity for error

  // Invariant 3: Best correct should be correct with respect to target
  assert(fxn(state.best_correct).first);
  // Invariant 4: Best yet should be less than or equal to correct cost
  assert(state.best_yet_cost <= state.current_cost);
}

Mcts::~Mcts(){
  delete_node(root_);
  assert(mcts_statistics_.num_nodes_ == 0);
}

} // namespace stoke
