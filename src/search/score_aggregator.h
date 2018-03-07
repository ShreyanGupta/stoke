#ifndef STOKE_SRC_SEARCH_SCORE_AGGREGATOR_H
#define STOKE_SRC_SEARCH_SCORE_AGGREGATOR_H

#include <utility>
#include <cmath>
#include <climits>

namespace stoke {

struct ScoreAggregator {
  ScoreAggregator() : score(0), score_sq(0), i(0) {}
  float score;
  double score_sq;
  int i;

  // Aggregate score
  // Modify this function to change the aggregation method
  virtual ScoreAggregator& operator+=(const float& s) = 0;

  // Return score
  virtual float get_score() = 0;

  // Return the avg and variance
  std::pair<float, float> get_statistics(){
    float avg = score/i;
    float var = score_sq/i - avg*avg;
    return std::make_pair(avg, var);
  }
};

struct AvgAggregator : public ScoreAggregator {
  
  ScoreAggregator& operator+=(const float& s){
    score += s;
    score_sq += s*s;
    ++i;
    return *this;
  }
  
  float get_score(){
    return score/i;
  }
};

struct MinAggregator : public ScoreAggregator {

  MinAggregator() : min_score(INT_MAX) {}
  float min_score;
  
  ScoreAggregator& operator+=(const float& s){
    min_score = std::min(min_score, s);
    score += s;
    score_sq += s*s;
    ++i;
    return *this;
  }
  
  float get_score(){
    return min_score;
  }
};

// Exponential Moving Average Aggregator
struct EMAAggregator : public ScoreAggregator {
  
  EMAAggregator(double gamma) : gamma(gamma), decay_score(0), weight(1) {
    gamma = std::min(gamma, 1.0);
  }
  
  double gamma;
  double decay_score;
  double weight;
  
  ScoreAggregator& operator+=(const float& s){
    decay_score += s*weight;
    weight *= gamma;
    score += s;
    score_sq += s*s;
    ++i;
    return *this;
  }
  
  float get_score(){
    return decay_score/i;
  }
};

} // namespace stoke

#endif