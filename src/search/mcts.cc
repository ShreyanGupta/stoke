#include <fstream>
#include <iostream>

#include <cassert>
#include <csignal>
#include <cmath>

#include "src/search/mcts.h"
#include "src/transform/weighted.h"

using namespace cpputil;
using namespace std;
using namespace std::chrono;
using namespace x64asm;

// TODO: Figure out about last_result_id in SearchState
// TODO: Figure out about the redo operation in Transform

namespace {

bool give_up_now = false;

void handler(int sig, siginfo_t* siginfo, void* context) {
  give_up_now = true;
}

void destructor_helper(stoke::Node* node){
  if(node == nullptr) return;
  for(auto* child : node->children){
    destructor_helper(child);
  }
  delete node;
}

void draw_graph_helper(std::ofstream& fout, stoke::Node* node){
  for(auto* child : node->children){
    fout << "N_" << node << " -> N_" << child << ";\n";
    draw_graph_helper(fout, child);
  }
  // fout << "N_" << node << " [label=\"" << node->score() << "/" << node->num_visit() << "\"];\n";
  fout << "N_" << node << " [label=\"\"];\n";
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

float Node::score(){
  // TODO: Update this to UCT
  // Need num_iteration for that
  return score_/num_visit_;
}


Mcts::Mcts(Transform* transform, int timeout_itr, int n, int r, int k) : 
  root_(new Node(nullptr)),
  n_(n), 
  r_(r), 
  k_(k), 
  transform_(transform),
  timeout_itr_(timeout_itr) {

  cout << "Enter MCTS constructor" << endl;
  set_seed(0);
  set_timeout_itr(0);
  set_timeout_sec(steady_clock::duration::zero());
  set_progress_callback(nullptr, nullptr);
  set_statistics_callback(nullptr, nullptr);
  set_statistics_interval(100000);

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


Node* Mcts::traverse(SearchState& state){
  Node* curr_node = root_;
  while(curr_node->children.size() != 0){
    size_t best_child_index = -1;
    float best_score = 0;
    auto& children = curr_node->children;
    auto& ti_vector = curr_node->ti_vector;
    assert(children.size() == ti_vector.size());
    for(size_t i=0; i<children.size(); ++i){
      auto* child = children[i];
      auto& ti = ti_vector[i];
      float score = child->score();
      if(score > best_score){
        best_score = score;
        best_child_index = i;
      }
    }
    curr_node = children[best_child_index];
    (*transform_).redo(state.current, ti_vector[best_child_index]);
  }
  return curr_node;
}

void Mcts::expand(Node* node, SearchState& state){
  auto& children = node->children;
  auto& ti_vector = node->ti_vector;
  children = std::vector<Node*>(k_);
  ti_vector = std::vector<TransformInfo>(k_);
  for(int i=0; i<k_; ++i){
    children[i] = new Node(node);
    while(!ti_vector[i].success){
      ti_vector[i] = (*transform_)(state.current);
    }
    (*transform_).undo(state.current, ti_vector[i]);
  }
}

float Mcts::rollout(Node* node, SearchState& state, CostFunction& fxn){
  float score = 0;
  // Rollout n times
  for(int i=0; i<n_; ++i){
    // Copy the init state
    SearchState curr_state = state;
    // Rollout for a depth of r
    for(int j=0; j<r_; ++j){
      // Random walk till depth r?

      // MCMC till depth r?
      TransformInfo ti = (*transform_)(curr_state.current);
      if(!ti.success) continue;

      const double p = prob_(gen_);
      const double max = curr_state.current_cost - (log(p) / beta_);

      const auto new_res = fxn(curr_state.current, max + 1);
      const bool is_correct = new_res.first;
      const auto new_cost = new_res.second;

      if(new_cost > max){
        (*transform_).undo(curr_state.current, ti);
        continue;
      }

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
      }

      // TODO: What do I put as the score?
      score += prob_(gen_);
    }

    // Update the global state
    update_global_state(curr_state, state);
  }
  // TODO: Decide what score to return for the node
  return score/n_/r_;
}

void Mcts::update(Node* node, float score){
  // Do no need SearchState to update
  Node* curr_node = node;
  while(curr_node != nullptr){
    curr_node->update(score);
    curr_node = curr_node->parent;
  }
}

void Mcts::run(const Cfg& target, CostFunction& fxn, Init init, SearchState& state, vector<TUnit>& aux_fxns){

  cout << "Enter MCTS run function" << endl;
  cout << "timeout_itr_ " << timeout_itr_ << " timeout_sec_ " << timeout_sec_.count() << endl;

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
    cout << "Iteration " << num_itr_ << endl;

    // When to exit the loop
    time_elapsed_ = duration_cast<duration<double>>(steady_clock::now() - start_time);
    if(
      num_itr_ >= timeout_itr_ ||
      (timeout_sec_ != steady_clock::duration::zero() && time_elapsed_ >= timeout_sec_) ||
      state.current_cost <= 0 ||
      give_up_now
    ) break;

    // Invoke statistics callback if we've been running for long enough
    if ((statistics_cb_ != nullptr) && (num_itr_ % interval_ == 0) && num_itr_ > 0) {
      statistics_cb_(get_statistics(), statistics_cb_arg_);
    }

    // Work with current_state
    SearchState curr_state = state;

    // Actual loop
    Node* leaf = traverse(curr_state);
    expand(leaf, curr_state);
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

  // draw_graph("graph.dot");
}

StatisticsCallbackData Mcts::get_statistics() const {
  return {move_statistics_, num_itr_, time_elapsed_, transform_};
}

void Mcts::stop() {
  give_up_now = true;
}

void Mcts::draw_graph(std::string file_name){
  std::ofstream fout(file_name);
  fout << "digraph G {\n";
  draw_graph_helper(fout, root_);
  fout << "}\n";
  fout.close();
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
  destructor_helper(root_);
}

} // namespace stoke
