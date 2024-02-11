mod def;
mod interactor;
mod util;

use crate::def::*;
use crate::interactor::*;
use crate::util::*;

use itertools::*;

#[macro_export]
#[cfg(not(feature = "local"))]
macro_rules! eprint {
    ($($_:tt)*) => {};
}

#[macro_export]
#[cfg(not(feature = "local"))]
macro_rules! eprintln {
    ($($_:tt)*) => {};
}

fn investigate(
    k: usize,
    query_count: usize,
    v: &Vec<Vec<usize>>,
    interactor: &mut Interactor,
    input: &Input,
) -> Vec<(Vec<(usize, usize)>, f64)> {
    const USE_HIGH_PROB: f64 = 0.5; // :param // NOTE: 徐々に大きくした方が良い
    let mut queries = Vec::with_capacity(query_count);

    let mut high_prob_v = vec![];
    if v.len() > 0 {
        for i in 0..input.n {
            for j in 0..input.n {
                let (pi, pj) = (i.max(1) - 1, j.max(1) - 1);
                let (ni, nj) = ((i + 1).min(input.n - 1), (j + 1).min(input.n - 1));
                if v[i][j]
                    + v[pi][j]
                    + v[i][pj]
                    + v[pi][pj]
                    + v[i][nj]
                    + v[ni][j]
                    + v[ni][nj]
                    + v[ni][pj]
                    + v[pi][nj]
                    > 0
                {
                    high_prob_v.push((i, j));
                }
            }
        }
    }

    for _ in 0..query_count {
        let mut s = vec![];
        while s.len() < k {
            let a = if high_prob_v.len() > 0 && rnd::nextf() < USE_HIGH_PROB {
                high_prob_v[rnd::gen_range(0, high_prob_v.len())]
            } else {
                (rnd::gen_range(0, input.n), rnd::gen_range(0, input.n))
            };
            if !s.contains(&a) {
                s.push(a);
            }
        }
        let obs_x = interactor.output_query(&s) as f64;
        let obs_x = ((obs_x - k as f64 * input.eps) / (1. - 2. * input.eps)).max(0.); // NOTE: 本当にあってる？
        queries.push((s, obs_x));
    }

    queries
}

struct MinoOptimizer<'a> {
    mino_range: Vec<(usize, usize)>,
    query_cache: Vec<f64>,
    query_indices: Vec<Vec<Vec<usize>>>,
    mino_pos: Vec<(usize, usize)>,
    score: f64,
    queries: &'a Vec<(Vec<(usize, usize)>, f64)>,
    input: &'a Input,
}

impl<'a> MinoOptimizer<'a> {
    fn new(queries: &'a Vec<(Vec<(usize, usize)>, f64)>, input: &'a Input) -> MinoOptimizer<'a> {
        let mino_range = get_mino_range(&input.minos);
        let mut mino_pos = Vec::with_capacity(input.m);

        // TODO: 初期解の改善
        for k in 0..input.m {
            mino_pos.push((
                rnd::gen_range(0, input.n - mino_range[k].0),
                rnd::gen_range(0, input.n - mino_range[k].1),
            ));
        }

        let v = get_v(&mino_pos, &input.minos, input.n);
        let mut query_cache = vec![0.; queries.len()];
        let mut query_indices = vec![vec![vec![]; input.n]; input.n];

        let mut score = 0.;
        for (q_i, (s, x)) in queries.iter().enumerate() {
            for &(i, j) in s.iter() {
                query_cache[q_i] += v[i][j] as f64;
                query_indices[i][j].push(q_i);
            }
            score += (query_cache[q_i] - x).powf(2.);
        }

        MinoOptimizer {
            mino_range,
            query_cache,
            query_indices,
            mino_pos,
            score,
            queries: &queries,
            input: &input,
        }
    }

    fn optimize(&mut self) {
        const ITERATION: usize = 10000; // :param
        for _t in 0..ITERATION {
            let mut score_diff = 0.;
            // let r = rnd::gen_range(2, 3.min(input.m) + 1); // :param
            let r = 2; // :param
            let sample_size = 10; // :param
            let cand_size = 3; // :param

            let mut mino_is = Vec::with_capacity(r);

            while mino_is.len() < r {
                let mino_i = rnd::gen_range(0, self.input.m);
                if mino_is.contains(&mino_i) {
                    continue;
                }
                mino_is.push(mino_i);
                score_diff += self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
            }
            let mut evals = vec![vec![]; r];
            for (i, &mino_i) in mino_is.iter().enumerate() {
                for _ in 0..sample_size {
                    let next_mino_pos = (
                        rnd::gen_range(0, self.input.n - self.mino_range[mino_i].0),
                        rnd::gen_range(0, self.input.n - self.mino_range[mino_i].1),
                    );
                    let eval = self.toggle_mino(mino_i, next_mino_pos, true);
                    evals[i].push((eval, next_mino_pos));
                    self.toggle_mino(mino_i, next_mino_pos, false);
                }
                evals[i].sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            let mut v = vec![0; r];
            if !self.dfs(&mut v, 0, cand_size, &mino_is, &evals, score_diff) {
                for mino_i in mino_is {
                    self.toggle_mino(mino_i, self.mino_pos[mino_i], true);
                }
            }
        }
    }

    fn dfs(
        &mut self,
        v: &mut Vec<usize>,
        depth: usize,
        cand_size: usize,
        mino_is: &Vec<usize>,
        evals: &Vec<Vec<(f64, (usize, usize))>>,
        score_diff: f64,
    ) -> bool {
        if depth == v.len() {
            let mut score_diff = score_diff;
            for i in 0..v.len() {
                score_diff += self.toggle_mino(mino_is[i], evals[i][v[i]].1, true);
            }
            let adopt = score_diff < -1e-6;
            if adopt {
                // eprintln!(
                //     "{:6} {:10.3} -> {:10.3} ({:10.3})",
                //     _t,
                //     self.score,
                //     self.score + score_diff,
                //     score_diff
                // );
                self.score += score_diff;
                for i in 0..v.len() {
                    self.mino_pos[mino_is[i]] = evals[i][v[i]].1;
                }
                return true;
            }
            for i in 0..v.len() {
                score_diff += self.toggle_mino(mino_is[i], evals[i][v[i]].1, false);
            }
            return false;
        }
        for i in 0..cand_size {
            v[depth] = i;
            if self.dfs(v, depth + 1, cand_size, mino_is, evals, score_diff) {
                return true;
            }
        }
        false
    }

    fn toggle_mino(&mut self, mino_i: usize, mino_pos: (usize, usize), turn_on: bool) -> f64 {
        let mut score_diff = 0.;
        for &(_i, _j) in self.input.minos[mino_i].iter() {
            let (i, j) = (mino_pos.0 + _i, mino_pos.1 + _j);
            for &q_i in self.query_indices[i][j].iter() {
                score_diff -= (self.query_cache[q_i] - self.queries[q_i].1).powf(2.);
                if turn_on {
                    self.query_cache[q_i] += 1.;
                } else {
                    self.query_cache[q_i] -= 1.;
                }
                score_diff += (self.query_cache[q_i] - self.queries[q_i].1).powf(2.);
            }
        }
        score_diff
    }

    // fn move_mino(
    //     &mut self,
    //     mino_i: usize,
    //     prev_mino_pos: (usize, usize),
    //     to_mino_pos: (usize, usize),
    // ) -> f64 {
    //     let mut score_diff = 0.;
    //     score_diff += self.toggle_mino(mino_i, prev_mino_pos, false);
    //     score_diff += self.toggle_mino(mino_i, to_mino_pos, true);
    //     score_diff
    // }
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n * input.n * 2;
    let k = input.n; // :param
    let query_count = input.n; // :param

    // 初期情報を集める
    let mut queries = investigate(k, query_count, &vec![], interactor, input);

    loop {
        // ミノの配置を最適化
        let mut optimizer = MinoOptimizer::new(&queries, &input);
        optimizer.optimize();
        eprintln!("optimize_mino_pos_error: {:10.5}", optimizer.score);

        let v = get_v(&optimizer.mino_pos, &input.minos, input.n);
        vis_v(&v, answer);

        eprintln!("error_count: {}", error_count(&v, answer));
        eprintln!("query_count: {} / {}", interactor.query_count, query_limit);
        eprintln!("total_cost:  {:.5}", interactor.total_cost);

        let s = get_s(&v);
        if interactor.output_answer(&s) {
            exit(interactor, input);
        }

        // 情報を集める
        let query_count = query_count.min(query_limit - interactor.query_count - 1);
        let _queries = investigate(k, query_count, &v, interactor, input);
        queries.extend(_queries);
    }
}

fn main() {
    time::start_clock();
    let mut interactor = Interactor::new();
    let input = interactor.read_input();
    let answer = if cfg!(feature = "local") {
        Some(interactor.read_answer(&input))
    } else {
        None
    };
    solve(&mut interactor, &input, &answer);
}
