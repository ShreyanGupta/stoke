// Copyright 2013-2016 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STOKE_SRC_SEARCH_SEARCH_H
#define STOKE_SRC_SEARCH_SEARCH_H

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
#include "src/tunit/tunit.h"

#include "src/search/mcts.h"

namespace stoke {

class Search {
public:
  /** Create a new search from a transform helper. */
  Search(Transform* transform) : mcts(Mcts(transform)) {}

  Search& set_mcts_args(int n, int r, int k, float exploration_factor){
    mcts.set_mcts_args(n,r,k,exploration_factor);
    return *this;
  }

  /** Set the random search seed. */
  Search& set_seed(std::default_random_engine::result_type seed) {
    mcts.set_seed(seed);
    return *this;
  }
  /** Set the maximum number of proposals to perform before giving up. */
  Search& set_timeout_itr(size_t timeout) {
    mcts.set_timeout_itr(timeout);
    return *this;
  }
  /** Set the maximum number of seconds to run for before giving up. */
  Search& set_timeout_sec(std::chrono::duration<double> timeout) {
    mcts.set_timeout_sec(timeout);
    return *this;
  }
  /** Set the annealing constant. */
  Search& set_beta(double beta) {
    mcts.set_beta(beta);
    return *this;
  }
  /** Set progress callback function. */
  Search& set_progress_callback(ProgressCallback cb, void* arg) {
    mcts.set_progress_callback(cb, arg);
    return *this;
  }
  /** Set new best correct callback function. */
  Search& set_new_best_correct_callback(NewBestCorrectCallback cb, void* arg) {
    mcts.set_new_best_correct_callback(cb, arg);
    return *this;
  }
  /** Set statistics callback function. */
  Search& set_statistics_callback(StatisticsCallback cb, void* arg) {
    mcts.set_statistics_callback(cb, arg);
    return *this;
  }
  /** Set the number of proposals to perform between statistics updates. */
  Search& set_statistics_interval(size_t si) {
    mcts.set_statistics_interval(si);
    return *this;
  }

  Search& set_mcts_statistics_interval(size_t si) {
    mcts.set_mcts_statistics_interval(si);
    return *this;
  }

  /** Run search beginning from a search state using a user-supplied cost function. */
  void run(const Cfg& target, CostFunction& fxn, Init init, SearchState& state, std::vector<stoke::TUnit>& aux_fxn) {
    mcts.run(target, fxn, init, state, aux_fxn);
  }
  /** Stops an in-progress search.  To be used from a callback, for example. */
  void stop(){
    mcts.stop();
  }

  /** Returns the statistics collected for the search up to now (or the full statistics for the whole run, if search terminated). */
  StatisticsCallbackData get_statistics() const {
    return mcts.get_statistics();
  }

private:
  Mcts mcts;
};

} // namespace stoke

#endif
