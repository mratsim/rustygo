// License
#[macro_use]
extern crate lazy_static;
extern crate fnv;
extern crate rand;
#[macro_use]
extern crate itertools;

use std::collections::HashMap;
// use std::collections::hash_state::DefaultState;
// use fnv::FnvHasher;
//Implemented pseudoRNG instead of FNVHaher:
//Random function is key to Monte-Carlo and should be optimized for AVX-128 SIMD.
//We don't need strong crypto so pseudoRNG is fine and much faster
use rand::{Rng, SeedableRng, Rand};
use std::num::Wrapping as w;
use std::ops::{Index, IndexMut};
use std::fmt;
use std::io::{stdin, BufRead};
use std::cmp::Ordering;
use std::ops::BitXor;

//Global param
const UCT_EXPLO_BIAS: f32 = 1.4142135623730951; //sqrt(2)
const MOVES_CUTOFF: u16 = 5;
const KOMI: f32 = 7.5; //Komi*2 to work only with integer
const MAX_SIMS: u16 = 200;
const SCORE_TO_BEAT: i16 = 361; //Number of Intersections (used for half-step counting)


//From Rust standard lib v1.6
/// Select an element from an iterator based on the given projection
/// and "comparison" function.
///
/// This is an idiosyncratic helper to try to factor out the
/// commonalities of {max,min}{,_by}. In particular, this avoids
/// having to implement optimizations several times.
#[inline]
fn select_fold1<I,B, FProj, FCmp>(mut it: I,
                                  mut f_proj: FProj,
                                  mut f_cmp: FCmp) -> Option<(B, I::Item)>
    where I: Iterator,
          FProj: FnMut(&I::Item) -> B,
          FCmp: FnMut(&B, &I::Item, &B, &I::Item) -> bool
{
    // start with the first element as our selection. This avoids
    // having to use `Option`s inside the loop, translating to a
    // sizeable performance gain (6x in one case).
    it.next().map(|mut sel| {
        let mut sel_p = f_proj(&sel);

        for x in it {
            let x_p = f_proj(&x);
            if f_cmp(&sel_p,  &sel, &x_p, &x) {
                sel = x;
                sel_p = x_p;
            }
        }
        (sel_p, sel)
    })
}

trait PartialMiniMax: Iterator {
//Currently max_by_key only works with Ord type, f64 only implements PartialOrd
//Only for f64/32 that can't NaN
#[inline]
    fn partial_max_by<B: PartialOrd, F>(self, f: F) -> Option<Self::Item>
        where Self: Sized, F: FnMut(&Self::Item) -> B,
    {
        select_fold1(self,
                     f,
                     // switch to y even if it is only equal, to preserve
                     // stability.
                     |x_p, _, y_p, _| x_p <= y_p)
            .map(|(_, x)| x)
    }
}

impl<I> PartialMiniMax for I where I: Iterator {}



// RNG XorShift128+ Implementation
// use extprim crate for u128 implementation?

#[allow(bad_style)]
type w64 = w<u64>;

pub struct XorShiftPlusRng {
    s: [w64; 2],
}

impl XorShiftPlusRng {
    pub fn new_unseeded() -> XorShiftPlusRng {
        XorShiftPlusRng { s: [w(0x193a6754a8a7d469), w(0x97830e05113ba7bb)] }
    }
}

impl Rng for XorShiftPlusRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s[0];
        let s0 = self.s[1];
        self.s[0] = s0;
        s1 = s1 ^ (s1 << 23); // a
        self.s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26); // b, c
        (self.s[1] + s0).0
    }
}

impl SeedableRng<[u64; 2]> for XorShiftPlusRng {
    /// Reseed an XorShiftPlusRng. This will panic if `seed` is entirely 0.
    fn reseed(&mut self, seed: [u64; 2]) {
        assert!(seed != [0, 0],
                "XorShiftPlusRng.reseed called with an all zero seed.");

        self.s = [w(seed[0]), w(seed[1])];
    }

    /// Create a new XorShiftPlusRng. This will panic if `seed` is entirely 0.
    fn from_seed(seed: [u64; 2]) -> XorShiftPlusRng {
        assert!(seed != [0, 0],
                "XorShiftPlusRng::from_seed called with an all zero seed.");

        XorShiftPlusRng { s: [w(seed[0]), w(seed[1])] }
    }
}

impl Rand for XorShiftPlusRng {
    fn rand<R: Rng>(rng: &mut R) -> XorShiftPlusRng {
        let mut seed: (u64, u64) = rng.gen();
        while seed == (0, 0) {
            seed = rng.gen();
        }
        XorShiftPlusRng { s: [w(seed.0), w(seed.1)] }
    }
}



// Use XorShift128 pseudoRNG due to speed
// macro_rules! hashmap {
//     ($( $key: expr => $val: expr ),*) => {{
//          let mut map = HashMap::with_hash_state(DefaultState::<FnvHasher>::default());
//          $( map.insert($key, $val); )*
//          map
//     	}}
// 	}

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = HashMap::new();
         $( map.insert($key, $val); )*
         map
    	}}
	}

#[derive(Debug,Copy,Clone,PartialEq,Eq,Hash)]
enum Intersection {
    Empty,
    Black,
    White,
    Border,
}

impl Default for Intersection {
    fn default() -> Intersection {
        Intersection::Empty
    }
}

impl fmt::Display for Intersection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Intersection::Empty => f.write_str("Empty"),
            Intersection::Black => f.write_str("Black"),
            Intersection::White => f.write_str("White"),
            Intersection::Border => f.write_str("Border"),
        }
    }
}

#[derive(Copy)]
struct Board19 {
    board: [Intersection; 21 * 21],
    bsize: u16,
    next_player: Intersection,  //Besoin?
}

impl Board19 {
    fn get_player(&self) -> &str {
        match self.next_player {
            Intersection::Black => "Black to play",
            Intersection::White => "White to play",
            _ => unreachable!(),
        }
    }
    fn flip_color(&self) -> Intersection {
        match self.next_player {
            Intersection::Black => Intersection::White,
            Intersection::White => Intersection::Black,
            _ => unreachable!(),
        }
    }

    #[inline]
    fn empty_inter(&self) -> Vec<usize> { //TODO return Iterator
        self.board
            .iter()
            .enumerate()
            .filter_map(|(idx, &item)| {
                match (idx, item) {
                    (_, Intersection::Empty) => Some(idx),
                    _ => None,
                }
            })
            .collect()
    }

    fn gen_nx_move(&self) -> Move {
        let mut rng: XorShiftPlusRng = rand::random();

        match rng.choose(&self.empty_inter()) {
            Some(idx) => Move::Coord {coord: *idx},
            None => Move::Pass,
        }
    }

    fn score_winner(&self) -> Intersection {
        let mut scoreboard = self.clone();

        while let Some(territory) = scoreboard.empty_inter().pop() {
            let touch_black = scoreboard.group(territory as u16)
                            .iter()
                            .flat_map(|&x| {
                                scoreboard.neighbors(x)
                                .iter()
                                .map(|&y| y)
                                .collect::<Vec<u16>>()
                            })
                            .any(|idx| *scoreboard.index(idx as usize)==Intersection::Black);
            let touch_white = scoreboard.group(territory as u16)
                            .iter()
                            .flat_map(|&x| {
                                scoreboard.neighbors(x)
                                .iter()
                                .map(|&y| y)
                                .collect::<Vec<u16>>()
                            })
                            .any(|idx| *scoreboard.index(idx as usize)==Intersection::White);

            let territory_gr = &scoreboard.group(territory as u16);
            match (touch_black, touch_white) {
                (true, false) =>  for i in territory_gr {   //TODO performance issue due to array bound checking?
                                    scoreboard[*i as usize] = Intersection::Black;
                                    },
                (false, true) =>  for i in territory_gr {
                                    scoreboard[*i as usize] = Intersection::White;
                                    },
                (true, true) =>   for i in territory_gr {
                                    scoreboard[*i as usize] = Intersection::Border;
                                    },
                _ => unreachable!(),
            }
        }

        let (black_score, white_score): (u16,u16) =
            scoreboard.board.iter().fold((0,0), |(black,white),&x| match x {
                                            Intersection::Black => (black+1,white),
                                            Intersection::White => (black,white+1),
                                            _ => (black,white),
                                        });
        //half-counting
        // let black_score: i16 =
        //     scoreboard.board.iter().fold(0, |black,&x| match x {
        //                                     Intersection::Black => black+2,
        //                                     _ => black,
        //                                 }) - DOUBLE_KOMI; //Warning negative overflow

        match PartialOrd::partial_cmp(&(black_score as f32),&(KOMI + (white_score as f32))) {
                    Some(Ordering::Less) => Intersection::White,
                    Some(Ordering::Greater) => Intersection::Black,
                    _ => unreachable!(),
        }
    }



}

impl Clone for Board19 {
    fn clone(&self) -> Self {
        *self
    }
}

impl Default for Board19 {
    fn default() -> Board19 {

        let mut b = [Intersection::Empty; 21 * 21];
        for i in 0..21 * 21 {
            // Side-effects
            if (i % 21 == 0) | (i % 21 == 20) | (i < 21) | (i > 21 * 20) {
                b[i] = Intersection::Border;
            }
        }
        Board19 {
            board: b,
            bsize: 21,
            next_player: Intersection::Black,
        }
    }
}

impl Index<usize> for Board19 {
    type Output = Intersection;
    fn index<'a>(&'a self, idx: usize) -> &'a Intersection {
        &self.board[idx]
    }
}


impl IndexMut<usize> for Board19 {
    fn index_mut<'a>(&'a mut self, idx: usize) -> &'a mut Intersection {
        self.board.index_mut(idx)
    }
}

impl fmt::Debug for Board19 {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.board[..].fmt(formatter)
    }
}

impl fmt::Display for Board19 {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let disp_mapping: HashMap<Intersection, char> = hashmap![ Intersection::Empty => '.',
															Intersection::Black => 'X',
															Intersection::White => 'O',
															Intersection::Border => '|'];
        let x: String = self.board
                            .iter()
                            .map(|&i: &Intersection| disp_mapping[&i])
                            .zip((0..21).cycle())
                            .fold(String::new(), |acc, (a, b)| {
                                    acc + &a.to_string() + if b == 20 {"\n"} else {""}
                            });
        write!(formatter, "{}", x)
    }
}

trait GroupLogic: Index<usize, Output=Intersection> {
    #[inline]
    fn neighbors(&self, pos: u16) -> [u16; 4];

    fn group(&self, pos: u16) -> Vec<u16> {
        let color = self.index(pos as usize);
        let mut candidates: Vec<u16> = vec![pos];
        let mut valid = Vec::<u16>::new();

        while let Some(candidate) = candidates.pop() {
            candidates.extend(
					self.neighbors(candidate).iter()
						.filter_map(|idx| match idx {
											_ if (|| valid.iter().any(|x| *x == *idx))() => None, //https://www.reddit.com/r/rust/comments/2ztiag/cannot_mutably_borrow_in_a_pattern_guard/
                                            _ if self.index(*idx as usize)==color => Some(idx),
                                            _ => None,
                                            })
				);
            valid.push(candidate);
        }
        valid
    }

    fn pseudo_lib(&self, pos: u16) -> u8 {
        self.group(pos)
            .iter()
            .flat_map(|&x| {
                self.neighbors(x)
                    .iter()
                    .map(|&y| y)
                    .collect::<Vec<u16>>()
            })
            .fold(0,
                  |acc, idx| acc + if *self.index(idx as usize) == Intersection::Empty {1} else {0}) //branching vs hashmap cost?
    }

    fn capture<'a>(&'a mut self, pos: u16) -> &'a mut Self;
	}

impl GroupLogic for Board19 {
    fn neighbors(&self, pos: u16) -> [u16; 4] {
        [pos - 1, pos - self.bsize, pos + 1, pos + self.bsize]
    }

    fn capture<'a>(&'a mut self, pos: u16) -> &'a mut Board19 {
        for i in self.group(pos) {
            self[i as usize] = Intersection::Empty;
        }
        self
    }
}

// Board coordinates do not have a I
fn board_setup(s: usize) -> (Board19, HashMap<String, usize>) {
    let textproduct = "aABCDEFGHJKLMNOPQRSTUVWXYZz".chars().take(s + 2);
    let boardsize = 0..s + 1 + 1;
    let coord_mapping: HashMap<_, _> = iproduct!(textproduct, boardsize)
                                           .map(|(a, b)| a.to_string() + &b.to_string())
                                           .zip(0..(s + 2).pow(2))
                                           .collect();
    let board: Board19 = Default::default();
    (board, coord_mapping)
}

#[derive(Debug,Copy,Clone)]
enum Move {
    Coord {coord: usize},
    Pass,
    Resign,
    TakeBack,
}

impl Move {
    fn translate(input: &str, coord_mapping: &HashMap<String, usize>) -> Option<Move> {
        match input {
            "Pass" => Some(Move::Pass),
            "Resign" => Some(Move::Resign),
            "Take back" => Some(Move::TakeBack),
            _ if coord_mapping.contains_key(input) => {
                Some(Move::Coord { coord: *coord_mapping.get(input).unwrap() })
            }
            _ => None,
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Move::Pass => formatter.write_str("Player passed"),
            Move::Resign => formatter.write_str("Player Resigned"),
            Move::TakeBack => formatter.write_str("Player took back a move"),
            Move::Coord {coord} => write!(formatter, "Move coordinates: {}", coord),
        }
    }
}


trait MoveCheck: GroupLogic{
    fn is_player_move_valid(&self, coord: usize) -> bool;
    fn moveplayer<'a>(&'a mut self, pos: usize) -> bool;
    fn movecore<'a>(&'a mut self, coord: usize);
}

impl MoveCheck for Board19 {
    #[inline]
    fn is_player_move_valid(&self, coord: usize) -> bool {
        // TODO reduce branching
        if self[coord] != Intersection::Empty {
            return false;
        }

        let mut test = self.to_owned();
        test[coord] = test.next_player;
        test.next_player = test.flip_color();

        // TODO: test if capture is valid and free some liberties
        if test.pseudo_lib(coord as u16) == 0 {
            return false;
        }
        // TODO: Hashing for Ko
        true
    }

    #[inline]
    fn movecore<'a>(&'a mut self, coord: usize){
        let mut c: Vec<u16> = self.neighbors(coord as u16)
                                  .into_iter()
                                  .filter(|&i| {
                                      self[*i as usize] == self.next_player &&
                                      self.pseudo_lib(*i) == 0
                                  })
                                  .map(|&x| x)
                                  .collect();

        while let Some(i) = c.pop() {
            self.capture(i);
        }
    }

    fn moveplayer<'a>(&'a mut self, coord: usize) -> bool {
        if self.is_player_move_valid(coord) {
            self[coord as usize] = self.next_player;
            self.next_player = self.flip_color();
        } else {
            println!("Invalid Move");
            return false;
        }
        self.movecore(coord);
        true
    }
}

#[derive(Debug,Clone)]
struct MCTS_tree{
    board_state: Board19,
    nx_player: Intersection,
    board_hash: u16,
    visits: u32,
    wins: u32,
    move_nb: u16,
    untried_moves: Vec<u16>,
    children: Vec<Zobrist>, // TODO: optim: VecDeque ?
    prev_nodes: Vec<Zobrist>,
    last_move: u16,
}

impl MCTS_tree {
    fn new() -> MCTS_tree{
        MCTS_tree {
            board_state: Default::default(),
            nx_player: Intersection::Empty,
            board_hash: 0,
            visits: 0,
            wins: 0,
            move_nb: 0,
            untried_moves: Vec::new(),
            children: Vec::new(),
            prev_nodes: Vec::new(),
            last_move: 0,
        }
    }

    //1. Selection step
    #[inline]
    fn uct(&self, &parent_sims: &u32) -> f32 {
        self.visits as f32 / self.wins as f32
        + UCT_EXPLO_BIAS * ((parent_sims as f32).ln() / (self.visits as f32)).sqrt()
    }

    fn mcts_select(&self, ref tt: &HashMap<Zobrist,MCTS_tree>) -> MCTS_tree {
        self.children.iter()
            .map(|x| tt.get(x).unwrap().clone())
            .partial_max_by(|tree| tree.uct(&self.visits)) //Beware of NaN if self.visits = 0
            .unwrap()
    }

    //2. Expansion step

fn expand(&self, tt: HashMap<Zobrist,MCTS_tree>) -> (MCTS_tree, HashMap<Zobrist,MCTS_tree>) {
        let mut rng: XorShiftPlusRng = rand::random();

        let mut nx_board = self.board_state.clone();
        let candidates = &nx_board.empty_inter();
                //test legality of move
                //test real eyes

        let candidate = rng.choose(candidates).unwrap();

        nx_board[*candidate] = nx_board.next_player;
        nx_board.next_player = nx_board.flip_color();

        let mut t = tt.clone();
        let hash = Zobrist::hash(self.board_state);

        t.get_mut(&hash).unwrap().children.push(Zobrist::hash(nx_board));

        let new_entry = MCTS_tree {
            board_state: nx_board,
            nx_player: nx_board.next_player,
            board_hash: 0,
            visits: 0,
            wins: 0,
            move_nb: self.move_nb+1,
            untried_moves: Vec::new(),
            children: Vec::new(),
            prev_nodes: vec![Zobrist::hash(self.board_state)],
            last_move: *candidate as u16,
        };


        t.insert(Zobrist::hash(nx_board),new_entry.clone());
        (new_entry,t)


    }

    //3. Simulate
    fn simulate(&self) -> Intersection {
        let mut simboard = self.board_state.clone();
        let mut n = self.move_nb;
        let mut passes = 0;

        while passes < 2 && n < MOVES_CUTOFF {
            match simboard.gen_nx_move() {
                Move::Pass => {passes+=1; continue;},
                Move::Coord {coord: idx} => {
                    simboard.movecore(idx);
                    passes=0;
                    n+=1;
                    },
                _ => unreachable!(),
            }
        }
        simboard.score_winner()
    }

    //4. Backpropagate -> Will be done in the main logic


    fn mcts_search(&self, th: Zobrist, tt: HashMap<Zobrist,MCTS_tree>) -> HashMap<Zobrist,MCTS_tree> {

        if self.board_state.empty_inter().len() == 0 { //TODO replace by legal move
            return tt;
        }

        let mut tree_hash = th.clone();
        let mut tree_descent: Vec<Zobrist> = vec![tree_hash];
        let mut temp_tree = self.clone();

        //Step 1, are there unexplored legal moves ?
        //If no descend until you find some
        println!("count of potential moves:\t{:?}", tt.get(&tree_hash).unwrap().board_state.empty_inter().len());
        println!("count of children moves:\t{:?}", tt.get(&tree_hash).unwrap().children.len());

        while tt.get(&tree_hash).unwrap().board_state.empty_inter().len() == 0 && tt.get(&tree_hash).unwrap().children.len() != 0 {
            println!("tree descent last move:\t{}",temp_tree.last_move);
            temp_tree = temp_tree.mcts_select(&tt);
            println!("tree descent selection:\t{}",temp_tree.last_move);
            tree_hash = Zobrist::hash(temp_tree.board_state);
            tree_descent.push(tree_hash);
            println!("tree descent list nodes:\t{}",temp_tree.last_move);
        }

        //Expand

        let (child, mut tt) = temp_tree.expand(tt);

        tree_descent.push(Zobrist::hash(child.board_state));

        let sim_result: Intersection = child.simulate();
        while let Some(z) = tree_descent.pop() {
             tt.get_mut(&z).unwrap().visits +=1;
             if sim_result==self.nx_player {
                 tt.get_mut(&z).unwrap().wins +=1;
             }
        }

        tt
    }

    fn mcts_controller(simu: u16, board: Board19) -> Move {
        let mut tree = MCTS_tree {
            board_state: board,
            nx_player: Intersection::White,
            board_hash: 0,
            visits: 0,
            wins: 0,
            move_nb: 0,
            untried_moves: Vec::new(),
            children: Vec::new(),
            prev_nodes: Vec::new(),
            last_move: 0,
        };

        let tree_hash = Zobrist::hash(board);
        let mut tt: HashMap<Zobrist,MCTS_tree> = hashmap!(tree_hash => tree.clone()); //Transpo table

        //Initialization on empty tree
        tt.get_mut(&tree_hash).unwrap().visits +=1;
        if tt.get(&tree_hash).unwrap().simulate()==tt.get(&tree_hash).unwrap().nx_player {
             tt.get_mut(&tree_hash).unwrap().wins +=1
        } //Manage Jigo

        let mut i: u16 = 0;
        while i < simu {
            tt = tree.mcts_search(tree_hash,tt);
            i+=1;
            println!("Iteration Monte-Carlo:\t{:?}", i);
            //println!("tt:\t{:?}", tt);
        }

        let nx_move = tt.get(&tree_hash)
                        .unwrap()
                        .children.iter()
                        .map(|x| tt.get(x).unwrap().clone())
                        .max_by_key(|tree| tree.visits)
                        .unwrap();

        println!("visits:\t{}\nwins:\t{}",nx_move.visits,nx_move.wins);
        println!("count children:\t{:?}", tt.get(&tree_hash)
                        .unwrap()
                        .children.len());

        Move::Coord {coord: nx_move.last_move as usize}

    }
}

// Struct to initialize each intersection to a random value
// First step in Zobrist hashing
struct CrossHash {
    random_hashes : [Zobrist; 21*21*2]
}

impl CrossHash {
    fn new() -> CrossHash {
        let mut arr = [Zobrist(0); 21*21*2];
        let mut rng: XorShiftPlusRng = rand::random();

        for i in arr.iter_mut() {
            *i = Zobrist(rng.gen());
        }

        CrossHash {random_hashes: arr}
    }
}

impl Index<usize> for CrossHash {
    type Output = Zobrist;
    fn index<'a>(&'a self, idx: usize) -> &'a Zobrist {
        &self.random_hashes[idx]
    }
}

lazy_static! {
    static ref SEED_HASHER: CrossHash = CrossHash::new();
}

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
struct Zobrist(u64);

impl BitXor for Zobrist {
    type Output = Zobrist;

    fn bitxor(self, _rhs: Zobrist) -> Zobrist {
        Zobrist(self.0 ^ _rhs.0)
    }
}

impl Zobrist{
    fn new() -> Zobrist {
        Zobrist(0)
    }

    fn hash_move(idx_move: usize,color: Intersection) -> Zobrist {
        let i = match color {
            Intersection::Black => 0,
            Intersection::White => 1,
            _ => unreachable!(),
        };
        SEED_HASHER[i * 21*21 + idx_move]
    }

    fn hash(board: Board19) -> Zobrist {
        //Beware of collision due to passes, use [Zobrist,Intersection] array as key?
        //&self should be a hash seed randomly initialized

        board.board.iter()
                    .enumerate()
                    .filter(|&(_,&color)| color == Intersection::Black || color == Intersection::White)
                    .fold(Zobrist::new(),|acc,(idx_move,&color)| acc ^ Zobrist::hash_move(idx_move,color))
    }

}




fn main() {

    println!("Hasher: {:?}", SEED_HASHER[1]);
    let mut board = board_setup(19);

    loop {
        println!("\n");
        println!("{:}", board.0);

        print!("\n{}\n", board.0.get_player());
        let mut h = Zobrist::hash(board.0);
        println!("Board hash {:?}",h);
        println!("\n");
        println!("Next move? ");

        // Read input
        let mut input = String::new();
        stdin()
            .read_line(&mut input)
            .ok()
            .expect("Failed to read line.");

        let trimmed = input.trim();
        if trimmed == "exit" {
            return;
        }

        let play = Move::translate(&trimmed, &board.1);
        match play {
            Some(Move::Coord {coord}) => {
                h = h ^ Zobrist::hash_move(coord, Intersection::Black);
                if !board.0.moveplayer(coord) {
                    continue;
                }} //Modify to capture group and check illegal moves
            _ => {
                println!("Oops! I didn't understand\n");
                continue;
            }
        }
        println!("Delta hash {:?}",h);
        println!("Full hash {:?}",Zobrist::hash(board.0));
        println!("\nCPU to move\n");

        let cpu_move = MCTS_tree::mcts_controller(MAX_SIMS,board.0); //replace by MonteCarlo
        match cpu_move {
            Move::Coord {coord} => {
                h = h ^ Zobrist::hash_move(coord, Intersection::White);
                if !board.0.moveplayer(coord) {
                    continue;
                }}
            _ => {
                println!("CPU did something not supported yet");
                continue;
            }
        }

        println!("Delta hash {:?}",h);
        println!("Full hash {:?}",Zobrist::hash(board.0));

    }
}
