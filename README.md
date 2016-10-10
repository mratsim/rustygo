# RustyGo, my untrusted bot player project

<!-- TOC -->

- [RustyGo, my untrusted bot player project](#rustygo-my-untrusted-bot-player-project)
    - [Why a bot in Rust](#why-a-bot-in-rust)
    - [History of computer go breakthrough](#history-of-computer-go-breakthrough)
    - [State](#state)
    - [Interesting repos](#interesting-repos)
    - [Future](#future)
        - [Roadmap:](#roadmap)
        - [Huge brain dump alert: optimization to think of in no particular order](#huge-brain-dump-alert-optimization-to-think-of-in-no-particular-order)
            - [Compilation:](#compilation)
            - [Vectorization](#vectorization)
            - [Localization / Branch Prediction / Cache](#localization-branch-prediction-cache)
            - [Randomization / MCTS optimization](#randomization-mcts-optimization)
            - [Data structure research](#data-structure-research)
            - [Machine Learning](#machine-learning)
            - [Algorithms](#algorithms)
            - [Consumption (of trees and iterators/vectors)](#consumption-of-trees-and-iteratorsvectors)
            - [Parallelism](#parallelism)
            - [Hashing](#hashing)
            - [Heuristics](#heuristics)
            - [Memory](#memory)
            - [Unit tests](#unit-tests)

<!-- /TOC -->

## Why a bot in Rust
I'm an amateur dan player hovering between 1d-3d on KGS and European Go ranking.
I've been interested in Computer Go since I started back in 2004 and followed eagerly the breakthroughs in the past 10 years.

Back in October 2015, I had the sudden urge to program a go bot.
* A bot in Haskell? Interesting but Monads and Functors gave me a headache.
	Also Haskell is reasonably fast but I had a lot of memory overflow with just prime computations so I don't think it will cut it with storing thousands of game trees.
* A bot in C/C++? I don't know C or C++ and it's already been done.
	Plus I like programming in functional style.

What is important for the bot:

* Memory, no leak, fast, compact: Go needs lot of memory operations with lots of possible moves, and board states of up to 400 moves to save in memory.
* Lock-free or at least concurrent data structure: One tree of variations, multiple threads that update it
* Access to neural networks libraries

So I choose Rust:

* The lure of CUDA/OpenCL libraries (Arrayfire, collenchyma)
* The lure of machine learning libraries (Leaf, rusty-machine)
* The lure of lock-free concurrent datastructure (crossbeam, rayon)
* a strong type system
* nice pattern matching (not as nice as Haskell though)
* function chaining without dozens of parenthesis
	if only I didn't have to .collect::<Vec<u16>>() every time
* no Garbage Collector, control of memory layout
* no memory leak (very important)
* Memory on the stack by default (the fastest(TM) )
* fast ~C like
* Good debugguer (LLVM)
* Unit testing included
* A strong and helpful community
* Does the coffee
If only it didn't have all those brackets

Other languague considered:
* Nim, high performance, very nice syntax, strong typing


## History of computer go breakthrough

1. First was the expert systems, with handcoded logic of "good moves"
	Too bad those expert system, collapsed in fights and against silly moves the programmer didn't expect.
2. Second was the statistics system, embodied by the Monte-Carlo Tree Search (MCTS) revolution.
	Before each move the go program simulate thousands of games and choose the one with best win ratio
	Several evolutions brought them up to dan level on 9x9 on 2008 CPUs :
		* UCT : which balances the Monte Carlo Tree exploration of known nodes vs new nodes
		* RAVE / AMAF : if a move is good later, consider it good also as the first move
		* Integration of heuristics and patterns that influence the Monte-Carlo Tree Search
	Thanks to the statistical approach, programs were much stronger in fights.
3. Third was the DCNNs (Deep Convolutional Neural Network)
	Those are used in combination with MCTS to direct the search to the most promising moves.


## State
RustyGo was started back in October and is currently compilable but buggy as hell, it might as well play random moves.
Also everything is in one file because I was in "hack away" mode and didn't learn how to split a projects into modules.

## Interesting repos
* Easy to dive in
	[Michi](https://github.com/pasky/michi) (Python) 
	[Pony-mcts](https://github.com/Mononofu/pony-mcts) (Pony and Rust)

* MCTS only
	[Libego](https://github.com/lukaszlew/libego) (C)

* Rusty:
	[Iomrascalai](Rust)

* "State of the art"
	[Darkforest](https://github.com/facebookresearch/darkforestGo) from Facebook Research, using Torch DCNN (Lua and C)
	[Pachi](https://github.com/pasky/pachi) long-time reference open-source engine before Facebook open-sourced theirs, with early Caffe DCNN support! (C)
	[Fuego](https://sourceforge.net/projects/fuego/) slightly weaker than pachi but very strong engine as well (C++)


## Future

### Roadmap:
* Find time to work on it
* Modularize the project
* Improve the data structure for concurrency, lock-free
* Improve the light playouts (no heuristics) to the fastest speed (cf pony-mcts and libego)
* Implement heuristics (ladder, local move preference) for heavy playouts
* Create a DCNN policy network to suggest moves
* ...
* ...
 
### Huge brain dump alert: optimization to think of in no particular order
#### Compilation:
* cargo rustc --release -- -C target-cpu=native i.e. exploit latest SIMD
* LLVM [Polly](http://polly.llvm.org/) - Polyhedral Optimization

#### Vectorization
AVX-512 board size

#### Localization / Branch Prediction / Cache
* Reduce Cache Misses
* Bitboards (needs AVX-512)
* Memoization
* HashMap instead of Match/if to reduce branching
* reduce if statement inside loop or fold. Use map and loop/fold after
* Auto sort MCTS tree according to UCT value (heap tree)
* SSE instruction or Xor or a+b-|a-b| or BLAS or [collenchyma](https://github.com/autumnai/collenchyma)/[Arrayfire](https://github.com/arrayfire/arrayfire-rust) to fold compare max(UCT) and reduce branching

#### Randomization / MCTS optimization
* UCT
* RAVE, AMAF (All moves as first)
* Fast number Gen (rdrand Hardware RNG ?)
* [CLOP](https://www.remi-coulom.fr/CLOP/) tuning (Confident Local Optimization for Noisy Black-Box Parameter Tuning)
* [Simulation Balancing](https://www.remi-coulom.fr/CG2010-Simulation-Balancing/SimulationBalancing.pdf) (cf Erica bot)
* FNV Hashmap algorithm (when it's possible to use a custom hasher)

#### Data structure research
Need fast delete of Early Nodes, traversal from leaf to start, very fast append, lock-free, parallelizable, cacheable, low memory use, max of O(1) operations
Board? Fixed Size Struct? Array by macro? Integer parametrized struct for different board size (not possible yet)

Some interesting things to explore
* [Trie](https://en.wikipedia.org/wiki/Trie)
* [Heap Tree](https://en.wikipedia.org/wiki/Binary_heap)
* [Finger Trees](http://scienceblogs.com/goodmath/2010/04/26/finger-trees-done-right-i-hope/)
* [Splay trees](https://en.wikipedia.org/wiki/Splay_tree)
* [Treaps](https://en.wikipedia.org/wiki/Treap)
* [Conc-Tree](https://en.wikipedia.org/wiki/Conc-Tree_list)
* Transpo Table with [Judy Array](https://en.wikipedia.org/wiki/Judy_array)
* Lock free hashmaps ([crossbeam](https://github.com/aturon/crossbeam))
* [Ctrie](https://en.wikipedia.org/wiki/Ctrie)
* [Disjoint Set data structure](https://en.wikipedia.org/wiki/Disjoint-set_data_structure) - Union/Find


Optimization that can help
* Pointer to child with max UCT value (need to separate in Controller data structure and other trees)
* Pointer to last parent in the tree for backpropagation
* Do not store a list/Vec of board state but a list of hash with a hash table.

Other stuff
* Lock-Free MCTS
* Efficient replacement of Hashmap and HashSet
* Remove all occurences of SignedInteger
* Remove all floating points (komi)
* [Connected Component](https://en.wikipedia.org/wiki/Connected_component_(graph_theory))
* [(Enriched) Common Fate Graph](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2005-126.pdf)
* Add a extra color around the 19x19 to reduce the branching (i.e. if on the edge ...)
* [Planar CRF](http://www.cs.ubc.ca/~murphyk/nips07NetworkWorkshop/abstracts/kamenetsky.pdf) (Conditional Random Fields)
* VecDeque instead of Vec?
* Disable Rust Array bound checking


#### Machine Learning
* [Linear Classifier]()http://www.lidi.info.unlp.edu.ar/WorldComp2011-Mirror/ICA8504.pdf 
* Neural Network & DCNN, [link Facebook Research](https://arxiv.org/pdf/1511.06410.pdf)
* Genetic Algorithm / Reinforcement learning

#### Algorithms
* FloodFill (Recursive vs Queue vs Stack)
* Blob Extraction
* Lambda search tree
* Analyze-after
* All Moves as first
* Grandparent Knowledge
* Experience tree
* nVidia: Aspiration Search, Principal Variation Search, Iterative Deepening
* Background History Reply Forest
* MCTS with Information Sharing
* Benson's Algorithm
* Partial Order Evaluation
* Last Good reply
* Linearly Decreasing Handicap Compensation
* Value-Based Situational Compensation
* Use symmetry

#### Consumption (of trees and iterators/vectors)
* Keeping Tree nodes
* Stream Fusion (for neighbor determination)
* Loop Fusion
* Use iterators instead of Vec for temporary methods/traits
* Implement Map+Fold/mapreduce fusion = mapAccumL et mapAccumR

#### Parallelism
* Multithreading
* GPU, GPGPU, Shader programming
* PTX/Cuda/OpenCL or ArrayFire GPU
* Leaf/Root/Tree Parallelization
* Multiblock parallel MCTS
* nVidia: Principal Variation Split, Young Brother Wait Concept, Dynamic Tree Splitting
* Timely dataflow
* Rust libraries: [Rayon](https://github.com/nikomatsakis/rayon), crossbeam, collenchyma

#### Hashing
Zobrist

#### Heuristics
* 2 liberties Semeai Solver
* Tsumego Solver
* Don't fill real eyes
* Negascout / Negamax
* Alpha Beta pruning
* Heat map / Computer Vision
* Futility pruning
* Razoring
* First/2nd lines heuristics (no if no stone around)
* Basic pattern shape
* Score counting
* Ladders
* Connection
* Legal moves: faire une passe toutes les x simulations opour leur donner une valeur null/négative

#### Memory
* use u8 everywhere (check usize)

#### Unit tests
Size of Intersection enum
* //println!("Size is {}", std::mem::size_of::<Intersection>());
Size of Board array
* //TODO: Need to profile which type is more efficient (stack vs heap, spacial locality, L1 and L2 cache ...)
* //Note: Dynamic size might be hard for compiler to optimize / vectorize

Board setup:
* //TODO: Unit test - make sure s^2+1 = 362 for 19x19 board size
* //TODO: Unit test : check for overflow (u8 -> 256)
* //TODO: Unit test - make sure numbers are properly cast as chars
    				*	// .inspect(|&x| println!("Coord:\t{:?}", x))

	* // TODO: Unit test - Check A1, A19, T1, T19 and neighbors
	* // TODO: Unit test - Check coordinates on 9x9 and 13x13

* //TODO: Check that after each move black & white are properly switching in memory
* //TODO: add unreachable
* //TODO: don't capture Intersection::Border