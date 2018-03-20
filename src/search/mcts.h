#ifndef STOKE_SRC_SEARCH_MCTS_H
#define STOKE_SRC_SEARCH_MCTS_H


#include <chrono>
#include <random>
#include <functional>
#include <utility>

#include "src/cost/cost_function.h"
#include "src/search/init.h"
#include "src/search/progress_callback.h"
#include "src/search/new_best_correct_callback.h"
#include "src/search/search_state.h"
#include "src/search/statistics.h"
#include "src/search/statistics_callback.h"
#include "src/search/mcts_statistics.h"
#include "src/search/mcts_node.h"
#include "src/transform/transform.h"
#include "src/tunit/tunit.h"

namespace stoke {

class Mcts { 
 public:
  Mcts(Transform* transform);
  ~Mcts();

  // Set the mcts arguments.
  Mcts& set_mcts_args(int n, int r, int k, float exploration_factor){
    n_ = n;
    r_ = r;
    k_ = k;
    exploration_factor_ = exploration_factor;
    return *this;
  }
  // Set the random search seed. 
  Mcts& set_seed(std::default_random_engine::result_type seed) {
    gen_.seed(seed);
    return *this;
  }
  // Set the maximum number of proposals to perform before giving up. 
  Mcts& set_timeout_itr(size_t timeout) {
    timeout_itr_ = timeout;
    return *this;
  }
  // Set the maximum number of seconds to run for before giving up. 
  Mcts& set_timeout_sec(std::chrono::duration<double> timeout) {
    timeout_sec_ = timeout;
    return *this;
  }
  // Set the annealing constant. 
  Mcts& set_beta(double beta) {
    beta_ = beta;
    return *this;
  }
  // Set progress callback function. 
  Mcts& set_progress_callback(ProgressCallback cb, void* arg) {
    progress_cb_ = std::bind(cb, std::placeholders::_1, arg);
    return *this;
  }
  // Set new best correct callback function. 
  Mcts& set_new_best_correct_callback(NewBestCorrectCallback cb, void* arg) {
    new_best_correct_cb_ = std::bind(cb, std::placeholders::_1, arg);
    return *this;
  }
  // Set statistics callback function. 
  Mcts& set_statistics_callback(StatisticsCallback cb, void* arg) {
    statistics_cb_ = std::bind(cb, std::placeholders::_1, arg);
    return *this;
  }
  // Set the number of proposals to perform between statistics updates. 
  Mcts& set_statistics_interval(size_t si) {
    statistics_interval_ = si;
    return *this;
  }
  // Set the number of proposals to perform between mcts statistics updates. 
  Mcts& set_mcts_statistics_interval(size_t si) {
    mcts_statistics_interval_ = si;
    return *this;
  }

  // Returns the statistics collected for the search up to now (or the full statistics for the whole run, if search terminated). 
  StatisticsCallbackData get_statistics() const;

  // Run search beginning from a search state using a user-supplied cost function. 
  void run(const Cfg& target, CostFunction& fxn, Init init, SearchState& state, std::vector<stoke::TUnit>& aux_fxn);
  // Stops an in-progress search.  To be used from a callback, for example. 
  void stop();

 private:
  // Not allowed to copy, assign
  Mcts& operator=(const Mcts& copy);

  Node* root_;
  int n_;  // Number of rollouts
  int r_;  // Depth of rollout
  int k_;  // Number of children of given node
  float exploration_factor_;  // Exploration vs exploitation factor

  Node* traverse(SearchState& state, int depth = -1);
  void expand(Node* node);
  float rollout(Node* node, SearchState& state, CostFunction& fxn);
  void update_node(Node* node, float score);

  float node_score(Node* node);
  // Returns (successful_transformation, is_correct)
  std::pair<bool,bool> mcmc_step(SearchState& state, CostFunction& fxn);
  // Update from local_state.current to state.best
  void update_state(SearchState& state, SearchState& local_state, bool is_correct);
  
  void trim(SearchState& state, int depth);
  void delete_node(Node* node, Node* new_root = nullptr);
  bool stop_now(bool force_stop);

  // Random generator. 
  std::default_random_engine gen_;
  // For sampling probabilities. 
  std::uniform_real_distribution<double> prob_;

  // Transformation helper class. 
  Transform* transform_;
  // Annealing constant. 
  double beta_;

  // How many iterations should search run for? 
  size_t timeout_itr_;
  // How many seconds should search run for? 
  std::chrono::duration<double> timeout_sec_;
  // Start time
  std::chrono::steady_clock::time_point start_time_;
  
  // Callbacks
  rProgressCallback progress_cb_;
  rNewBestCorrectCallback new_best_correct_cb_;
  rStatisticsCallback statistics_cb_;
  
  // How often are statistics printed? 
  size_t statistics_interval_;
  size_t mcts_statistics_interval_;

  // Statistics so far. 
  std::vector<Statistics> move_statistics_;
  size_t num_mcmc_itr_;
  std::chrono::duration<double> time_elapsed_;

  // MCTS statistics
  MctsStatistics mcts_statistics_;
  size_t num_itr_;

  // Configures a search state. 
  void configure(const Cfg& target, CostFunction& fxn, SearchState& state, std::vector<stoke::TUnit>& aux_fxn) const;

};

} // namespace stoke

#endif