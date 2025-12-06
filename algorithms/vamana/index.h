// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// #pragma once is a header guard that prevents this file from being included multiple times
// in a single compilation unit. It's a non-standard but widely supported alternative to
// traditional include guards (#ifndef/#define/#endif)
#pragma once

// #include is a preprocessor directive that inserts the contents of another file
// <math.h> is the C standard library for mathematical functions (sqrt, pow, etc.)
#include <math.h>

// Standard C++ library headers (using angle brackets means they're system/standard library headers)
// <algorithm> provides functions like std::sort, std::unique for working with sequences
#include <algorithm>
// <random> provides random number generation facilities (random_generator, distributions)
#include <random>
// <set> provides the std::set container, an ordered collection of unique elements
#include <set>

// Project-specific headers (using quotes means they're local to the project)
// "../utils/point_range.h" defines the PointRange abstraction for collections of points
#include "../utils/point_range.h"
// "../utils/graph.h" defines the Graph data structure for storing neighbor relationships
#include "../utils/graph.h"
// "../utils/types.h" defines common types like BuildParams, QueryParams, and stats
#include "../utils/types.h"
// "parlay/parallel.h" provides parallel programming primitives (parallel_for, etc.)
#include "parlay/parallel.h"
// "parlay/primitives.h" provides functional programming primitives (map, filter, reduce)
#include "parlay/primitives.h"
// "parlay/delayed.h" provides lazy evaluation utilities for delaying computation
#include "parlay/delayed.h"
// "parlay/random.h" provides parallel-safe random number generation
#include "parlay/random.h"
// "../utils/beamSearch.h" implements the beam search algorithm for graph traversal
#include "../utils/beamSearch.h"

// namespace declares a named scope to avoid naming conflicts with other code
// All code in this namespace must be accessed with parlayANN:: prefix (or using namespace)
namespace parlayANN {

// template<...> declares a template, which is a blueprint for creating classes/functions
// that work with different types. The typename keyword indicates type parameters.
// This creates a generic knn_index struct that can work with different point types and index types
template<typename PointRange, typename QPointRange, typename indexType>
// struct defines a structure (like class but members are public by default)
// knn_index is the Vamana/DiskANN k-nearest neighbor index data structure
struct knn_index {
  // using creates type aliases (shorthand names) for complex types
  // typename is required when accessing nested types in template parameters
  // PointRange::Point extracts the Point type from the PointRange template parameter
  using Point = typename PointRange::Point;
  // QPoint is the query point type (may differ from base points, e.g., for quantization)
  using QPoint = typename QPointRange::Point;
  // distanceType is the numeric type used for distances (e.g., float, double)
  // Point::distanceType extracts it from the Point class
  using distanceType = typename Point::distanceType;
  // pid is shorthand for "point id" - a pair of (index, distance)
  // std::pair<T1, T2> is a standard library template that holds two values
  using pid = std::pair<indexType, distanceType>;
  // PR and QPR are shorter aliases for the point range types
  using PR = PointRange;
  using QPR = QPointRange;
  // GraphI is the graph type templated on the index type
  // Graph<indexType> stores adjacency lists where each node is identified by indexType
  using GraphI = Graph<indexType>;

  // Member variables (data stored in each instance of this struct)
  // BuildParams is a struct containing parameters for building the index (R, L, alpha, etc.)
  BuildParams BP;
  // std::set<T> is an ordered collection of unique elements
  // delete_set tracks which points have been deleted (for dynamic index updates)
  std::set<indexType> delete_set;
  // start_point is the entry point for searches in the graph (usually node 0)
  indexType start_point;

  // Constructor: special function called when creating a knn_index object
  // Takes a reference (&) to BuildParams to avoid copying
  // : BP(BP) is the initializer list syntax, which initializes member BP with parameter BP
  knn_index(BuildParams &BP) : BP(BP) {}

  // Member function (method) that returns the start point for searches
  // indexType is the return type, get_start is the function name
  // { return start_point; } is the function body
  indexType get_start() { return start_point; }

  // robustPrune: Core algorithm from DiskANN paper for selecting diverse neighbors
  // This function prunes a candidate set to select R diverse neighbors that provide
  // good coverage for routing queries through the graph
  // Returns: std::pair of (pruned neighbor sequence, distance computation count)
  // std::pair<T1, T2> is a template that holds two values of potentially different types
  std::pair<parlay::sequence<indexType>, long>
  // Function parameters:
  // - indexType p: the point whose neighbors we're selecting
  // - parlay::sequence<pid>& cand: reference (&) to sequence of candidate neighbors with distances
  //   (& means we pass by reference to avoid copying the entire sequence)
  // - GraphI &G: reference to the graph structure containing current neighbor lists
  // - PR &Points: reference to the collection of all points for distance calculations
  // - double alpha: pruning parameter controlling edge diversity (typical: 1.0-1.2)
  // - bool add = true: optional parameter with default value, controls whether to include
  //   existing neighbors of p in the candidate set
  robustPrune(indexType p, parlay::sequence<pid>& cand,
              GraphI &G, PR &Points, double alpha, bool add = true) {
    // Add existing out-neighbors of p to the candidate set
    // size_t is an unsigned integer type for sizes/counts, typically 64-bit on modern systems
    // G[p] accesses the adjacency list for point p using operator[]
    // .size() returns the number of neighbors
    size_t out_size = G[p].size();
    // std::vector<T> is a dynamic array that can grow/shrink
    // Creates an empty vector to hold candidate (point_id, distance) pairs
    std::vector<pid> candidates;
    // long is a signed integer type (at least 32 bits) for counting distance computations
    long distance_comps = 0;
    // Range-based for loop: iterates over each element in cand
    // auto deduces the type automatically (here it's pid)
    // x is a copy of each element (we could use auto& for reference)
    for (auto x : cand) candidates.push_back(x);

    // if statement: conditional execution based on boolean expression
    // When add is true, include existing neighbors of p as additional candidates
    if(add){
      // Traditional for loop: for (initialization; condition; increment)
      // Iterates from i=0 to i<out_size
      for (size_t i=0; i<out_size; i++) {
        // Increment counter for each distance computation
        distance_comps++;
        // std::make_pair(a, b) constructs a std::pair object
        // G[p][i] gets the i-th neighbor of p (accessing nested containers with [][])
        // Points[G[p][i]].distance(Points[p]) computes distance between neighbor and p
        // .push_back() adds element to end of vector
        candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
      }
    }

    // Sort the candidate set in ascending order of distance from p
    // Lambda function syntax: [capture](parameters) { body }
    // [&] captures all local variables by reference (can access candidates, etc.)
    // auto deduces the return type (bool in this case)
    // This comparator sorts first by distance (a.second), breaking ties by index (a.first)
    auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b) {
      // .second accesses the distance in the pair, .first accesses the index
      // || is logical OR, && is logical AND
      // Returns true if a should come before b in sorted order
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    // std::sort sorts the range [begin, end) using the comparator function
    // .begin() returns iterator to first element, .end() returns iterator past last element
    std::sort(candidates.begin(), candidates.end(), less);

    // Remove any duplicate point IDs from the sorted candidate list
    // std::unique moves duplicates to the end and returns iterator to new logical end
    // Lambda [&] (auto x, auto y) defines equality: two candidates are equal if IDs match
    // auto deduces parameter types (here they're pid = std::pair<indexType, distanceType>)
    auto new_end =std::unique(candidates.begin(), candidates.end(),
			      [&] (auto x, auto y) {return x.first == y.first;});
    // Construct a new vector from the unique elements only
    // std::vector(iterator_begin, iterator_end) constructs from a range
    candidates = std::vector(candidates.begin(), new_end);

    // Create vector to store the pruned neighbor list (will contain at most R neighbors)
    std::vector<indexType> new_nbhs;
    // .reserve(n) pre-allocates memory for n elements to avoid reallocations
    // BP.R is the degree bound (maximum number of neighbors)
    new_nbhs.reserve(BP.R);

    // Index to track current position in the sorted candidates list
    size_t candidate_idx = 0;

    // Main pruning loop: select up to R diverse neighbors
    // while loop continues as long as both conditions are true (&&)
    // .size() returns the current number of elements in new_nbhs
    while (new_nbhs.size() < BP.R && candidate_idx < candidates.size()) {
      // Get the next candidate point (p_star) from the sorted list
      // .first accesses the point ID from the pair
      int p_star = candidates[candidate_idx].first;
      // Move to next candidate for next iteration (post-increment)
      candidate_idx++;
      // Skip if candidate is the point itself or marked as invalid (-1)
      // continue skips to next iteration of the loop
      // == is equality comparison, || is logical OR
      if (p_star == p || p_star == -1) {
        continue;
      }

      // Add p_star to the neighbor list (it's a good diverse neighbor)
      new_nbhs.push_back(p_star);

      // Prune candidates that are too close to p_star (they would form long triangle edges)
      // This ensures diversity: we don't want all neighbors clustered together
      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        // Get the candidate point p_prime
        int p_prime = candidates[i].first;
        // Only process valid candidates (not marked as -1)
        if (p_prime != -1) {
          // Count this distance computation
          distance_comps++;
          // Compute distance between the newly added neighbor (p_star) and candidate (p_prime)
          distanceType dist_starprime = Points[p_star].distance(Points[p_prime]);
          // Get the distance from p to p_prime (already computed and stored)
          // .second accesses the distance value in the pair
          distanceType dist_pprime = candidates[i].second;
          // RNG inequality test: if dist(p_star, p_prime) * alpha <= dist(p, p_prime)
          // then p_prime is too close to p_star and doesn't add diversity
          // Mark it as invalid (-1) so it won't be selected later
          // alpha controls pruning aggressiveness (typically 1.0-1.2)
          if (alpha * dist_starprime <= dist_pprime) {
            // Mark this candidate as pruned by setting its ID to -1
            candidates[i].first = -1;
          }
        }
      }
    }

    // Convert std::vector to parlay::sequence (parallel sequence type)
    // parlay::to_sequence creates a parallel sequence from standard container
    auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
    // return statement with std::pair constructor
    // Returns both the pruned neighbor list and the distance computation count
    return std::pair(new_neighbors_seq, distance_comps);
  } // End of robustPrune function

  // Overloaded version of robustPrune: same function name but different parameters
  // This wrapper accepts candidates without precomputed distances and computes them
  // Function overloading allows multiple functions with same name but different signatures
  // Returns: std::pair of (pruned neighbors, total distance computations)
  std::pair<parlay::sequence<indexType>, long>
  // Parameters: similar to above but candidates is sequence of IDs only (no distances)
  robustPrune(indexType p, parlay::sequence<indexType> candidates,
              GraphI &G, PR &Points, double alpha, bool add = true){

    // Create a sequence to hold candidates with computed distances
    // parlay::sequence<T> is a parallel-aware dynamic array
    parlay::sequence<pid> cc;
    // Track distance computations performed in this function
    long distance_comps = 0;
    // Pre-allocate space for efficiency
    // .size() returns number of elements in candidates sequence
    cc.reserve(candidates.size()); // + size_of(p->out_nbh));
    // Loop through candidates and compute distance to p for each
    // ++i is pre-increment (slightly more efficient than i++)
    for (size_t i=0; i<candidates.size(); ++i) {
      // Count this distance computation
      distance_comps++;
      // Compute distance and create (id, distance) pair
      // candidates[i] is the point ID, Points[candidates[i]] is the point object
      // .distance(Points[p]) calls the distance method on the point
      cc.push_back(std::make_pair(candidates[i], Points[candidates[i]].distance(Points[p])));
    }
    // Structured binding (C++17): auto [a, b] = pair; unpacks the pair into two variables
    // Calls the first robustPrune overload with distances included
    auto [ngh_seq, dc] = robustPrune(p, cc, G, Points, alpha, add);
    // Return combined results: neighbor sequence and total distance computations
    // dc is from the first robustPrune call, distance_comps is from this wrapper
    return std::pair(ngh_seq, dc + distance_comps);
  } // End of robustPrune wrapper

  // Helper function: add neighbors from ngh to candidates, avoiding duplicates
  // template<typename...> makes this function work with different container types
  // const rangeType1& means we won't modify ngh (const) and pass by reference (&)
  // rangeType2& means we pass candidates by reference so we can modify it
  template<typename rangeType1, typename rangeType2>
  // void means this function doesn't return a value
  void add_neighbors_without_repeats(const rangeType1 &ngh, rangeType2& candidates) {
    // std::unordered_set<T> is a hash set providing O(1) average lookup time
    // Used to track which elements are already in candidates
    std::unordered_set<indexType> a;
    // Range-based for loop: iterate through candidates and add each to the set
    // auto deduces type, for (element : container) syntax
    for (auto c : candidates) a.insert(c);
    // Loop through neighbors in ngh
    // Traditional for loop with index i
    for (int i=0; i < ngh.size(); i++)
      // .count(key) returns 1 if key exists in set, 0 otherwise
      // Add to candidates only if not already present (count == 0)
      if (a.count(ngh[i]) == 0) candidates.push_back(ngh[i]);
  } // End of add_neighbors_without_repeats

  // Simple setter function: initializes the start point to 0
  // The start_point is the entry node for beam search (typically the first point added)
  // Curly braces {} define the function body
  void set_start(){start_point = 0;}

  // Main function to build the Vamana graph index
  // Takes references (&) to graph, points, query points, statistics, and optional sort flag
  void build_index(GraphI &G, PR &Points, QPR &QPoints,
                   stats<indexType> &BuildStats, bool sort_neighbors = true){
    // std::cout is output stream for printing to console
    // << is stream insertion operator, std::endl adds newline and flushes buffer
    std::cout << "Building graph..." << std::endl;
    // Initialize the start point (entry node for searches)
    set_start();
    // Create a sequence of all point indices to insert
    // parlay::tabulate(n, f) creates sequence of length n where element i = f(i)
    // Lambda [&] (size_t i) captures variables by reference and takes index i
    // static_cast<T>(value) converts value to type T (here size_t to indexType)
    parlay::sequence<indexType> inserts = parlay::tabulate(Points.size(), [&] (size_t i){
      return static_cast<indexType>(i);});
    // Check if using single-batch mode (insert all points at once with random initial edges)
    // != 0 checks if BP.single_batch is non-zero (true in C++)
    if (BP.single_batch != 0) {
      // degree stores the number of random initial edges per node
      int degree = BP.single_batch;
      // Print message about the initialization strategy
      std::cout << "Using single batch per round with " << degree << " random start edges" << std::endl;
      // parlay::random_generator provides parallel-safe random number generation
      parlay::random_generator gen;
      // std::uniform_int_distribution<long> generates random integers uniformly in [0, G.size())
      // Template parameter <long> specifies the integer type
      std::uniform_int_distribution<long> dis(0, G.size());
      // parlay::parallel_for(start, end, func) executes func(i) in parallel for i in [start, end)
      // Lambda [&] (long i) captures local variables by reference, takes node index i
      parlay::parallel_for(0, G.size(), [&] (long i) {
        // Create vector to hold random neighbor IDs for this node
        // std::vector<indexType>(degree) constructs vector with degree elements
        std::vector<indexType> outEdges(degree);
        // Generate degree random neighbors for node i
        for (int j = 0; j < degree; j++) {
          // gen[seed] gets a random number generator for specific seed
          // i*degree + j ensures different seeds for different (i,j) pairs
          auto r = gen[i*degree + j];
          // dis(r) generates random integer using generator r
          outEdges[j] = dis(r);
        }
        // Update node i's neighbor list with the random edges
        // G[i] accesses node i, .update_neighbors() sets its adjacency list
        G[i].update_neighbors(outEdges);
      });
    }

    // Build the graph in multiple passes (refinement iterations)
    // The last pass uses the specified alpha parameter for more aggressive pruning
    std::cout << "number of passes = " << BP.num_passes << std::endl;
    // Loop through each pass of the graph construction
    for (int i=0; i < BP.num_passes; i++) {
      // Check if this is the last pass (i == BP.num_passes - 1)
      if (i == BP.num_passes - 1)
        // Last pass: use specified alpha for final pruning
        // Parameters: inserts, graph, points, query points, stats, alpha, random_order, base, max_fraction
        batch_insert(inserts, G, Points, QPoints, BuildStats, BP.alpha, true, 2, .02);
      else
        // Earlier passes: use alpha=1.0 (no aggressive pruning) for faster convergence
        batch_insert(inserts, G, Points, QPoints, BuildStats, 1.0, true, 2, .02);
    }

    // Optionally sort each node's neighbors by distance (improves query performance)
    // sort_neighbors is a parameter with default value true
    if (sort_neighbors) {
      // Parallel loop over all nodes in the graph
      // Space after parlay::parallel_for is optional (formatting variation)
      parlay::parallel_for (0, G.size(), [&] (long i) {
        // Lambda defining comparison function: j < k if distance(i,j) < distance(i,k)
        // Sorts neighbors of i by their distance to i
        auto less = [&] (indexType j, indexType k) {
          return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
        // G[i].sort(comparator) sorts node i's neighbor list using the comparator
        G[i].sort(less);});
    }
  } // End of build_index

  // Batch insertion: core function for inserting points into the graph
  // Uses exponentially growing batch sizes for efficiency
  // Parameters:
  // - inserts: sequence of point indices to insert
  // - G: the graph being built
  // - Points, QPoints: base and query point collections
  // - BuildStats: statistics tracker for distance computations
  // - alpha: pruning parameter (typically 1.0-1.2)
  // - random_order: whether to shuffle insertion order (default false)
  // - base: exponential growth factor for batch sizes (default 2)
  // - max_fraction: maximum batch size as fraction of n (default 0.02 = 2%)
  // - print: whether to print progress messages (default true)
  void batch_insert(parlay::sequence<indexType> &inserts,
                    GraphI &G, PR &Points, QPR &QPoints,
                    stats<indexType> &BuildStats, double alpha,
                    bool random_order = false, double base = 2,
                    double max_fraction = .02, bool print=true) {
    // Validate all point indices before insertion
    // Range-based for: iterates through each point p in inserts
    for(int p : inserts){
      // Check if point index is valid (within graph bounds)
      // static_cast<int>(x) converts unsigned size_t to signed int for comparison
      if(p < 0 || p > (int) G.size()){
        // Print error message if invalid index found
        std::cout << "ERROR: invalid point "
                  << p << " given to batch_insert" << std::endl;
        // abort() terminates the program immediately
        abort();
      }
    }
    // n is total number of points in the graph
    size_t n = G.size();
    // m is number of points to insert in this batch_insert call
    size_t m = inserts.size();
    // inc tracks which exponential batch we're on (batch size = base^inc)
    size_t inc = 0;
    // count tracks total number of points inserted so far
    size_t count = 0;
    // frac tracks progress as fraction of n (for progress reporting)
    float frac = 0.0;
    // Print progress every 10% (0.1)
    float progress_inc = .1;
    // Calculate maximum batch size: either max_fraction * n or 1 million, whichever is smaller
    // std::min(a, b) returns the smaller of two values
    // static_cast converts between numeric types (here: double to size_t, int to float)
    // 1000000ul is unsigned long literal (ul suffix)
    size_t max_batch_size = std::min(static_cast<size_t>(max_fraction * static_cast<float>(n)),
                                     1000000ul);
    // Safety check: ensure max_batch_size is at least 1
    // If max_fraction * n rounds down to 0, use n instead
    if(max_batch_size == 0) max_batch_size = n;
    // Create permutation for randomizing insertion order
    // parlay::sequence<T> is a parallel-aware dynamic array of type T
    parlay::sequence<int> rperm;
    // If random_order is requested, create random permutation
    if (random_order) 
      // parlay::random_permutation<T>(n) creates sequence [0..n-1] in random order
      // Template parameter <int> specifies the element type
      rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else
      // Otherwise create identity permutation [0, 1, 2, ..., m-1]
      // parlay::tabulate(n, f) creates sequence where element i = f(i)
      // Lambda [&](int i) { return i; } just returns the index unchanged
      rperm = parlay::tabulate(m, [&](int i) { return i; });
    // Apply permutation to inserts to get shuffled sequence
    // For each position i, take the element at position rperm[i]
    // auto deduces the type (here parlay::sequence<indexType>)
    auto shuffled_inserts =
      parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });
    // Create timers to measure performance of each phase
    // parlay::internal::timer is a timing utility for benchmarking
    // String parameter is the timer name for reporting
    parlay::internal::timer t_beam("beam search time");
    parlay::internal::timer t_bidirect("bidirect time");
    parlay::internal::timer t_prune("prune time");
    // Initially stop all timers (they start when constructed)
    // .stop() pauses the timer
    t_beam.stop();
    t_bidirect.stop();
    t_prune.stop();
    // Main loop: process points in batches of exponentially increasing size
    // while loop continues until all m points have been processed
    while (count < m) {
      // floor: starting index of current batch (inclusive)
      // ceiling: ending index of current batch (exclusive)
      size_t floor;
      size_t ceiling;
      // Check if we're still in exponential growth phase
      // pow(base, inc) computes base^inc (e.g., 2^0=1, 2^1=2, 2^2=4, ...)
      if (pow(base, inc) <= max_batch_size) {
        // Exponential growth: batch size = base^(inc+1) - base^inc
        // floor is one less than base^inc (since we start at 0)
        // E.g., for base=2: batch 0 has floor=0, batch 1 has floor=1, batch 2 has floor=3
        floor = static_cast<size_t>(pow(base, inc)) - 1;
        // ceiling is one less than base^(inc+1), capped at m
        // std::min(a, b) returns smaller value
        ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
        // Update count to track total points processed
        count = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
      } else {
        // Linear growth phase: once batches reach max_batch_size, stop growing exponentially
        // Each batch has fixed size of max_batch_size
        floor = count;
        ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
        count += static_cast<size_t>(max_batch_size);
      }

      // Special case: if single_batch mode, insert all points at once
      if (BP.single_batch != 0) {
        floor = 0;
        ceiling = m;
        count = m;
      }

      // Create storage for new neighbor lists for points in this batch
      // parlay::sequence<parlay::sequence<indexType>> is a 2D array: each element is a sequence
      // Size is (ceiling-floor) = number of points in this batch
      parlay::sequence<parlay::sequence<indexType>> new_out_(ceiling-floor);
      // Phase 1: Beam search to find candidate neighbors for each point
      // For each point, do beam search from start_point and use visited nodes as candidates
      // Start timing the beam search phase
      t_beam.start();

      // Process each point in the current batch in parallel
      // parlay::parallel_for(start, end, func) runs func(i) for each i in [start, end)
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        // Get actual point index from shuffled sequence
        size_t index = shuffled_inserts[i];
        // Determine starting point for beam search
        // Ternary operator: condition ? value_if_true : value_if_false
        // In single_batch mode, each point starts from itself (i), otherwise use global start_point
        int sp = BP.single_batch ? i : start_point;
        // Create query parameters for beam search
        // QueryParams constructor: (cut, beamSize, k, visited_limit, degree_limit)
        // Cast to long using (long) to match expected type
        // 0 for cut (no pruning), BP.L for beam width, 0.0 for unused param
        QueryParams QP((long) 0, BP.L, (double) 0.0, (long) Points.size(), (long) G.max_degree());
        // Perform beam search and get visited nodes + distance computation count
        // auto [a, b] = func() is structured binding: unpacks returned pair into two variables
        auto [visited, bs_distance_comps] =
          // beam_search_rerank__ is the beam search implementation with reranking
          // Template parameters: <Point, QPoint, PR, QPR, indexType> specify types
          // Arguments: point to insert, query point, graph, point collections, start point, params
          //beam_search<Point, PointRange, indexType>(Points[index], G, Points, sp, QP);
          beam_search_rerank__<Point, QPoint, PR, QPR, indexType>(Points[index],
                                                                 QPoints[index],
                                                                 G,
                                                                 Points,
                                                                 QPoints,
                                                                 sp,
                                                                 QP);
        // Update statistics: track distance computations for this point
        // BuildStats.increment_dist increments the distance computation counter
        BuildStats.increment_dist(index, bs_distance_comps);
        // Track number of nodes visited during beam search
        // .size() returns number of elements in visited sequence
        BuildStats.increment_visited(index, visited.size());

        // Prune the visited set to select R diverse neighbors
        // long is the type for storing distance computation count
        long rp_distance_comps;
        // std::tie(a, b) = func() is older C++ syntax for unpacking pairs (pre-C++17)
        // Assigns first element to new_out_[i-floor], second to rp_distance_comps
        // i-floor converts global index i to batch-local index
        std::tie(new_out_[i-floor], rp_distance_comps) = robustPrune(index, visited, G, Points, alpha);
        // Update statistics with pruning distance computations
        BuildStats.increment_dist(index, rp_distance_comps);
      });

      // Update the graph with computed neighbor lists
      // Parallel loop over batch indices
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        // Update point's neighbors in the graph
        // G[shuffled_inserts[i]] accesses the node, .update_neighbors() sets its adjacency list
        // new_out_[i-floor] is the computed neighbor list for this point
        G[shuffled_inserts[i]].update_neighbors(new_out_[i-floor]);
      });

      // Stop beam search timer (beam search + pruning phase complete)
      t_beam.stop();

      // Phase 2: Make edges bidirectional
      // For each edge (p, q) added, we also need to add reverse edge (q, p)
      // Start timing the bidirectional edge phase
      t_bidirect.start();

      // Create flattened list of (neighbor, source) pairs for all new edges
      // parlay::delayed operations use lazy evaluation (computed only when needed)
      // parlay::delayed::flatten concatenates sequences of sequences into single sequence
      auto flattened = parlay::delayed::flatten(parlay::tabulate(ceiling - floor, [&](size_t i) {
        // Get the source point index
        indexType index = shuffled_inserts[i + floor];
        // For each neighbor of this point, create pair (neighbor, source)
        // parlay::delayed::map(seq, f) applies function f to each element of seq lazily
        // [=] captures variables by value (copies index into the lambda)
        // new_out_[i] is the sequence of neighbors for point at position i
        return parlay::delayed::map(new_out_[i], [=] (indexType ngh) {
          // Create pair where key is neighbor, value is source
          // This represents reverse edge: neighbor should have source as a neighbor
          return std::pair(ngh, index);});}));
      // Group all pairs by key (neighbor ID) to collect all reverse edges for each node
      // parlay::group_by_key groups pairs with same key into (key, sequence_of_values)
      // parlay::delayed::to_sequence converts lazy delayed sequence to actual sequence
      auto grouped_by = parlay::group_by_key(parlay::delayed::to_sequence(flattened));

      // Stop bidirectional phase timer
      t_bidirect.stop();
      // Phase 3: Add reverse edges and prune if necessary
      // Start timing the pruning phase
      t_prune.start();
      // For each node that received incoming edges, add them to its neighbor list
      // If adding edges exceeds degree bound R, prune using robustPrune
      // Parallel loop over all nodes that received new incoming edges
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        // Structured binding with reference (&): unpacks grouped_by[j] without copying
        // auto& means we get references to avoid copying large sequences
        // index is the node receiving incoming edges, candidates are the sources
        auto &[index, candidates] = grouped_by[j];
        // Calculate new total degree if we add all candidates
        // G[index].size() is current degree, candidates.size() is number of new edges
	size_t newsize = candidates.size() + G[index].size();
        // Check if adding all candidates stays within degree bound
        // <= is less than or equal to operator
        if (newsize <= BP.R) {
          // Degree bound not exceeded: add candidates without pruning
	  // First add existing neighbors to candidates to avoid duplicates
	  add_neighbors_without_repeats(G[index], candidates);
          // Update graph with combined neighbor list
	  G[index].update_neighbors(candidates);
        } else {
          // Degree bound exceeded: use robustPrune to select best R neighbors
          // std::move(candidates) transfers ownership, avoiding copy (move semantics)
          // Structured binding unpacks the returned pair
          auto [new_out_2_, distance_comps] = robustPrune(index, std::move(candidates), G, Points, alpha);
          // Update graph with pruned neighbor list
	  G[index].update_neighbors(new_out_2_);
          // Update statistics with distance computations from pruning
          BuildStats.increment_dist(index, distance_comps);
        }
      });
      // Stop pruning phase timer
      t_prune.stop();

      // Print progress updates
      // && is logical AND: both conditions must be true
      // Only print if print flag is true and not in single_batch mode
      if (print && BP.single_batch == 0) {
        // Calculate index threshold for next progress message
        // auto deduces type (here float)
        auto ind = frac * n;
        // Check if this batch crosses the progress threshold
        // floor <= ind < ceiling means we just passed the threshold
        if (floor <= ind && ceiling > ind) {
          // Increment progress fraction (move to next 10% milestone)
          frac += progress_inc;
          // Print progress percentage
          // 100 * frac converts fraction to percentage
          std::cout << "Pass " << 100 * frac << "% complete"
                    << std::endl;
        }
      }
      // Increment batch counter for next iteration
      // += 1 is equivalent to inc = inc + 1 or inc++
      inc += 1;
    } // End of while loop (all batches processed)
    // Print total time for each phase
    // .total() prints accumulated time from all start/stop cycles
    t_beam.total();
    t_bidirect.total();
    t_prune.total();
  } // End of batch_insert function

}; // End of knn_index struct

} // end namespace parlayANN
