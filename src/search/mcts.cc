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

bool force_stop = false;

void handler(int sig, siginfo_t* siginfo, void* context) {
  force_stop = true;
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

Mcts::Mcts(Transform* transform) : 
  root_(new Node(nullptr)),
  transform_(transform), 
  num_mcmc_itr_(0),
  mcts_statistics_(MctsStatistics(num_itr_, num_mcmc_itr_, time_elapsed_)) {

  set_seed(0);
  set_timeout_itr(0);
  set_timeout_sec(steady_clock::duration::zero());
  set_progress_callback(nullptr, nullptr);
  set_statistics_callback(nullptr, nullptr);
  set_statistics_interval(100000);
  set_mcts_args(1, 1000, 4, 1);

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

    for(size_t i=0; i<children.size(); ++i){
      float score = node_score(children[i]);
      // Pick the lowest score
      if(score < least_score){
        least_score = score;
        best_child_index = i;
      }
    }
    assert(best_child_index != -1);
    curr_node = children[best_child_index];
    ++curr_depth;
  }
  state->current = curr_node->cfg;
  state->current_cost = curr_node->cost;
  return curr_node;
}

void Mcts::expand(Node* node){
  auto& children = node->children;
  assert(children.size() == 0);
  for(int i=0; i<k_; ++i){
    children.push_back(new Node(node));
    mcts_statistics_.num_nodes_++;
  }
}

float Mcts::rollout(Node* node, SearchState& state, CostFunction& fxn){
  // Rollout n times
  for(int i=0; i<n_; ++i){
    SearchState curr_state = state;
    // Rollout for a depth of r
    for(int j=0; j<r_; ++j){
      if(stop_now(force_stop)) break;
      auto result = mcmc_step(curr_state, fxn);
      if(result.first) update_state(state, curr_state, result.second);
    }
  }
  node->cfg = state.best_yet;
  node->cost = state.best_yet_cost;
  return node->cost;
}

void Mcts::update_node(Node* node, float score){
  // Do no need SearchState to update
  Node* curr_node = node;
  while(curr_node != nullptr){
    curr_node->update_node(score);
    curr_node = curr_node->parent;
  }
}

float Mcts::node_score(Node* node){
  if(node->num_visit_ == 0) return 0;

  float x = node->score_ / node->num_visit_;
  float confidence = sqrt(2*log(num_itr_ + 1) / node->num_visit_);  
  // -ve confidence as we are choosing the least value
  return x - exploration_factor_ * confidence;
}

pair<bool,bool> Mcts::mcmc_step(SearchState& state, CostFunction& fxn){
  num_mcmc_itr_++;
  if ((statistics_cb_ != nullptr) && (num_mcmc_itr_ % statistics_interval_ == 0)) {
    statistics_cb_(get_statistics());
  }
  
  TransformInfo ti = (*transform_)(state.current);
  move_statistics_[ti.move_type].num_proposed++;
  if(!ti.success) return make_pair(false, false);
  
  move_statistics_[ti.move_type].num_succeeded++;

  const double p = prob_(gen_);
  const double max = state.current_cost - (log(p) / beta_);

  const auto new_res = fxn(state.current, max + 1);
  const bool is_correct = new_res.first;
  const auto new_cost = new_res.second;

  if(new_cost > max){
    (*transform_).undo(state.current, ti);
    return make_pair(false, is_correct);
  }
  move_statistics_[ti.move_type].num_accepted++;
  state.current_cost = new_cost;
  return make_pair(true, is_correct);
}

void Mcts::update_state(SearchState& state, SearchState& local_state, bool is_correct){
  auto& new_cost = local_state.current_cost;

  bool new_best_yet = new_cost < state.best_yet_cost;
  bool new_best_correct_yet = is_correct && ((new_cost == 0) || (new_cost < state.best_correct_cost));
  
  if (new_best_yet) {
    state.best_yet = local_state.current;
    state.best_yet_cost = new_cost;
  }
  
  if (new_best_correct_yet) {
    state.success = true;
    state.best_correct = local_state.current;
    state.best_correct_cost = new_cost;
  }

  if(new_best_correct_cb_ != nullptr && new_best_correct_yet){
    new_best_correct_cb_({state});
  }

  if ((progress_cb_ != nullptr) && (new_best_yet || new_best_correct_yet)) {
    progress_cb_({state});
    mcts_statistics_.time_cost_vec_.push_back(make_pair(time_elapsed_.count(), new_cost));
  }
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

bool Mcts::stop_now(bool force_stop){
  time_elapsed_ = duration_cast<duration<double>>(steady_clock::now() - start_time_);
  return (
    force_stop ||
    num_itr_ >= timeout_itr_ ||
    (timeout_sec_ != steady_clock::duration::zero() && time_elapsed_ >= timeout_sec_)
  );
}

void Mcts::run(const Cfg& target, CostFunction& fxn, Init init, SearchState& state, vector<TUnit>& aux_fxns){

  // Configure initial state
  configure(target, fxn, state, aux_fxns);

  // Make sure target and rewrite are sound to begin with
  assert(state.best_yet.is_sound());
  assert(state.best_correct.is_sound());

  // Statistics callback variables. earch only works with 'WeightedTransform'
  move_statistics_ = vector<Statistics>(static_cast<WeightedTransform*>(transform_)->size());
  start_time_ = chrono::steady_clock::now();

  // Early corner case bailouts
  if (state.current_cost == 0) {
    state.success = true;
    state.best_correct = state.current;
    state.best_correct_cost = 0;
    return;
  }

  force_stop = false;

  for(num_itr_ = 0; true; ++num_itr_){
    // When to exit the loop
    if(stop_now(force_stop)) break;

    // Work with current_state
    SearchState curr_state = state;
    Node* leaf = traverse(curr_state);
    
    // Invoke statistics callback if we've been running for long enough
    if(num_itr_ % mcts_statistics_interval_ == 0){
      mcts_statistics_.print(leaf);
    }

    expand(leaf);
    for(auto* child : leaf->children){
      float score = rollout(child, curr_state, fxn);
      update_node(child, score);
    }

    // Update global state
    update_global_state(curr_state, state);

  } // End of loop

  if (force_stop) {
    state.interrupted = true;
  }

  // make sure Cfg's are in a valid state (e.g. liveness information, which we
  // do not update during search)
  state.current.recompute();
  state.best_correct.recompute();
  state.best_yet.recompute();

  SearchState curr_state = state;
  mcts_statistics_.print(traverse(curr_state));
}

StatisticsCallbackData Mcts::get_statistics() const {
  return {move_statistics_, num_mcmc_itr_, time_elapsed_, transform_};
}

void Mcts::stop() {
  force_stop = true;
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
