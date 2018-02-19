#include <cassert>
#include <csignal>
#include <cmath>

#include "src/search/mcts.h"
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

void destructor_helper(Node* node){
  if(node == nullptr) return;
  for(auto* child : node->children){
    destructor_helper(child);
  }
  delete node;
}

void draw_graph_helper(std::ofstream& fout, Node* node){
  for(auto* child : node->children){
    fout << "N_" << node << " -> N_" << child << ";\n";
    draw_graph_helper(fout, child);
  }
  // fout << "N_" << node << " [label=\"" << node->score() << "/" << node->num_visit() << "\"];\n";
  fout << "N_" << node << " [label=\"\"];\n";
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
  transform_(transform),
  timeout_itr_(timeout_itr),
  n_(n), 
  r_(r), 
  k_(k), 
  root_(new Node(nullptr)) {

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


Node* Mcts::traverse(){
  Node* curr_node = root_;
  while(curr_node->children.size() != 0){
    Node* best_node = nullptr;
    float best_score = 0;
    for(auto* child : curr_node->children){
      float score = child->score();
      if(score > best_score){
        best_score = score;
        best_node = child;
      }
    }
    curr_node = best_node;
  }
  return curr_node;
}

void Mcts::expand(Node* node){
  auto& children = node->children;
  children = std::vector<Node*>(k_);
  for(int i=0; i<k_; ++i){
    children[i] = new Node(node);
  }
}

float Mcts::rollout(Node* node){
  float score = 0;
  // Rollout n times
  for(int i=0; i<n_; ++i){
    // Rollout for a depth of r
    for(int j=0; j<r_; ++j){
      // TODO: Rollout here
      score += rand_(gen_);
    }
  }
  // TODO: Decide what score to return
  return score/n_/r_;
}

void Mcts::update(Node* node, float score){
  Node* curr_node = node;
  while(curr_node != nullptr){
    curr_node->update(score);
    curr_node = curr_node->parent;
  }
}

void Mcts::run(const Cfg& target, CostFunction& fxn, Init init, SearchState& state, vector<TUnit>& aux_fxns){

  // Basic asserts
  assert(timeout_itr_ > 0);
  assert(timeout_sec_ != steady_clock::duration::zero());

  // Configure initial state
  configure(target, fxn, state, aux_fxns);

  // Make sure target and rewrite are sound to begin with
  assert(state.best_yet.is_sound());
  assert(state.best_correct.is_sound());

  // Statistics callback variables
  // FIXME: Search only works with 'WeightedTransform', because it needs
  // statistics.
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
    time_elapsed_ = duration_cast<duration<double>>(steady_clock::now() - start);
    if(
      num_itr_ >= timeout_itr_ ||
      time_elapsed_ >= timeout_sec_ ||
      state.current_cost <= 0 ||
      give_up_now
    ) break;

    // Invoke statistics callback if we've been running for long enough
    if ((statistics_cb_ != nullptr) && (num_itr_ % interval_ == 0) && num_itr_ > 0) {
      statistics_cb_(get_statistics(), statistics_cb_arg_);
    }

    // Actual loop
    cout << "Iteration " << num_itr_ << endl;
    Node* leaf = traverse();
    expand(leaf);
    for(auto* child : leaf->children){
      float score = rollout(child);
      update(child, score);
    }
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

StatisticsCallbackData Search::get_statistics() const {
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
