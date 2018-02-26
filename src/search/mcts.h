#ifndef STOKE_SRC_SEARCH_MCTS_H
#define STOKE_SRC_SEARCH_MCTS_H


#include <chrono>
#include <random>

#include "src/cost/cost_function.h"
#include "src/search/init.h"
#include "src/search/progress_callback.h"
#include "src/search/new_best_correct_callback.h"
#include "src/search/search_state.h"
#include "src/search/statistics.h"
#include "src/search/statistics_callback.h"
#include "src/transform/transform.h"
#include "src/transform/info.h"
#include "src/tunit/tunit.h"

namespace stoke {

class Node {
 private:
  int num_visit_;
  float score_;

 public:
  Node(Node* parent) : num_visit_(0), score_(0), parent(parent) {}
  Node* parent;
  // std::vector<std::pair<Node*, TransformInfo> > children
  std::vector<Node*> children;
  std::vector<TransformInfo> ti_vector;
  
  // Function to update score
  void update(float score);

  // Function to get score of node
  float score();
};


class Mcts { 
 public:
  Mcts(Transform* transform, int timeout_itr, int n, int r, int k);
  ~Mcts();
  // Visualizing graph
  void draw_graph(std::string file_name);

  // All functions from original Search class below

  /** Set the random search seed. */
  Mcts& set_seed(std::default_random_engine::result_type seed) {
    gen_.seed(seed);
    return *this;
  }
  /** Set the maximum number of proposals to perform before giving up. */
  Mcts& set_timeout_itr(size_t timeout) {
    timeout_itr_ = timeout;
    return *this;
  }
  /** Set the maximum number of seconds to run for before giving up. */
  Mcts& set_timeout_sec(std::chrono::duration<double> timeout) {
    timeout_sec_ = timeout;
    return *this;
  }
  /** Set the annealing constant. */
  // Don't need it for MCTS
  Mcts& set_beta(double beta) {
    beta_ = beta;
    return *this;
  }
  /** Set progress callback function. */
  Mcts& set_progress_callback(ProgressCallback cb, void* arg) {
    progress_cb_ = cb;
    progress_cb_arg_ = arg;
    return *this;
  }
  /** Set new best correct callback function. */
  Mcts& set_new_best_correct_callback(NewBestCorrectCallback cb, void* arg) {
    new_best_correct_cb_ = cb;
    new_best_correct_cb_arg_ = arg;
    return *this;
  }
  /** Set statistics callback function. */
  Mcts& set_statistics_callback(StatisticsCallback cb, void* arg) {
    statistics_cb_ = cb;
    statistics_cb_arg_ = arg;
    return *this;
  }
  /** Set the number of proposals to perform between statistics updates. */
  Mcts& set_statistics_interval(size_t si) {
    interval_ = si;
    return *this;
  }

  /** Returns the statistics collected for the search up to now (or the full statistics for the whole run, if search terminated). */
  StatisticsCallbackData get_statistics() const;

  /** Run search beginning from a search state using a user-supplied cost function. */
  void run(const Cfg& target, CostFunction& fxn, Init init, SearchState& state, std::vector<stoke::TUnit>& aux_fxn);
  /** Stops an in-progress search.  To be used from a callback, for example. */
  void stop();

 private:
  // Not allowed to copy
  Mcts& operator=(const Mcts& copy);

  Node* root_;
  // int timeout_itr_;
  int n_;  // Number of rollouts
  int r_;  // Depth of rollout
  int k_;  // Number of children

  std::uniform_real_distribution<float> rand_;

  Node* traverse(SearchState& state);
  void expand(Node* node, SearchState& state);
  float rollout(Node* node, SearchState& state, CostFunction& fxn);
  void update(Node* node, float score);
  

  // All original private members

  /** Random generator. */
  std::default_random_engine gen_;
  /** For sampling moves. */
  // std::uniform_int_distribution<size_t> int_;
  /** For sampling probabilities. */
  std::uniform_real_distribution<double> prob_;

  /** Transformation helper class. */
  Transform* transform_;

  /** How many iterations should search run for? */
  size_t timeout_itr_;
  /** How many seconds should search run for? */
  std::chrono::duration<double> timeout_sec_;
  /** Annealing constant. */
  double beta_;

  /** Progress callback. */
  ProgressCallback progress_cb_;
  void* progress_cb_arg_;
  /** New best correct callback. */
  NewBestCorrectCallback new_best_correct_cb_;
  void* new_best_correct_cb_arg_;
  /** Statistics callback. */
  StatisticsCallback statistics_cb_;
  void* statistics_cb_arg_;
  /** How often are statistics printed? */
  size_t interval_;

  /** Statistics so far. */
  std::vector<Statistics> move_statistics_;
  size_t num_itr_;
  std::chrono::duration<double> time_elapsed_;

  /** Configures a search state. */
  void configure(const Cfg& target, CostFunction& fxn, SearchState& state, std::vector<stoke::TUnit>& aux_fxn) const;

};

} // namespace stoke

#endif