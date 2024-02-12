mod def;
mod interactor;
mod util;

use crate::def::*;
use crate::interactor::*;
use crate::util::*;

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

const EPS: f64 = 1e-6;
const D: [(usize, usize); 8] = [
    (0, 1),
    (1, 0),
    (0, !0),
    (!0, 0),
    (1, 1),
    (!0, 1),
    (!0, !0),
    (1, !0),
];

fn create_weighted_delta(delta_max_dist: i64) -> Vec<(i64, i64)> {
    let mut delta = vec![];
    let p = 2.; // :param
    for di in -delta_max_dist..=delta_max_dist {
        for dj in -delta_max_dist..=delta_max_dist {
            let dist = ((i64::abs(di) + i64::abs(dj)) as f64).max(1.);
            let cnt = ((delta_max_dist as f64 * 2.).powf(p) / dist.powf(p))
                .round()
                .max(1.) as usize;
            delta.extend(vec![(di, dj); cnt]);
        }
    }
    delta
}

fn add_delta(
    from_pos: (usize, usize),
    mino_range: (usize, usize),
    delta: (i64, i64),
) -> (usize, usize) {
    // NOTE: 外れているならNoneを返す、現状は少し偏っている
    let ni = (from_pos.0 as i64 + delta.0).clamp(0, mino_range.0 as i64 - 1) as usize;
    let nj = (from_pos.1 as i64 + delta.1).clamp(0, mino_range.1 as i64 - 1) as usize;
    (ni, nj)
}

fn investigate(
    k: usize,
    query_count: usize,
    v: &Vec<Vec<usize>>,
    interactor: &mut Interactor,
    input: &Input,
) -> Vec<(Vec<(usize, usize)>, f64)> {
    const USE_HIGH_PROB: f64 = 0.5; // :param // NOTE: 徐々に大きくした方が良さそう
    let mut queries = Vec::with_capacity(query_count);

    let mut high_prob_v = vec![];
    if v.len() > 0 {
        for i in 0..input.n {
            for j in 0..input.n {
                for (di, dj) in D {
                    let (ni, nj) = (i + di, j + dj);
                    if ni < input.n && nj < input.n && v[ni][nj] > 0 {
                        high_prob_v.push((i, j));
                        break;
                    }
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
    adopt_count: usize,
    queries: &'a Vec<(Vec<(usize, usize)>, f64)>,
    input: &'a Input,
}

impl<'a> MinoOptimizer<'a> {
    fn new(queries: &'a Vec<(Vec<(usize, usize)>, f64)>, input: &'a Input) -> MinoOptimizer<'a> {
        let mino_range = get_mino_range(&input);
        let mut mino_pos = Vec::with_capacity(input.m);

        // TODO: 初期解の改善
        for k in 0..input.m {
            mino_pos.push((
                rnd::gen_range(0, mino_range[k].0),
                rnd::gen_range(0, mino_range[k].1),
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
            adopt_count: 0,
            queries: &queries,
            input: &input,
        }
    }

    fn optimize(&mut self, iteration: usize) {
        let delta_max_dist = 2; // :param
        let weighted_delta = create_weighted_delta(delta_max_dist);

        let mut mino_is = vec![];
        let mut next_mino_poss = vec![];

        for _t in 0..iteration {
            mino_is.clear();
            next_mino_poss.clear();

            let p = rnd::nextf();
            let score_diff = if p < 0.2 {
                self.action_slide(&mut mino_is, &mut next_mino_poss, 1, &weighted_delta)
            } else if p < 0.3 {
                self.action_move_one(&mut mino_is, &mut next_mino_poss)
            } else {
                self.action_swap(&mut mino_is, &mut next_mino_poss, 2, &weighted_delta)
            };

            let adopted = score_diff < -EPS;
            if adopted {
                self.score += score_diff;
                for i in 0..mino_is.len() {
                    self.mino_pos[mino_is[i]] = next_mino_poss[i];
                }
                self.adopt_count += 1;
            } else {
                for i in 0..mino_is.len() {
                    self.toggle_mino(mino_is[i], next_mino_poss[i], false);
                    self.toggle_mino(mino_is[i], self.mino_pos[mino_is[i]], true);
                }
            }
        }

        eprintln!("adopt_count: {} / {}", self.adopt_count, iteration);
    }

    fn action_swap(
        &mut self,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
        r: usize,
        weighted_delta: &Vec<(i64, i64)>,
    ) -> f64 {
        let r = r.min(self.input.m);
        let mut score_diff = 0.;
        while mino_is.len() < r {
            let mino_i = rnd::gen_range(0, self.input.m);
            if mino_is.contains(&mino_i) {
                continue;
            }
            mino_is.push(mino_i);
            score_diff += self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        }
        for i in 0..r {
            let delta = weighted_delta[rnd::gen_range(0, weighted_delta.len())];
            let next_mino_pos = add_delta(
                self.mino_pos[mino_is[(i + 1) % r]],
                self.mino_range[mino_is[i]],
                delta,
            );
            score_diff += self.toggle_mino(mino_is[i], next_mino_pos, true);
            next_mino_poss.push(next_mino_pos);
        }

        score_diff
    }

    fn action_slide(
        &mut self,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
        r: usize,
        weighted_delta: &Vec<(i64, i64)>,
    ) -> f64 {
        let r = r.min(self.input.m);
        let mut score_diff = 0.;
        while mino_is.len() < r {
            let mino_i = rnd::gen_range(0, self.input.m);
            if mino_is.contains(&mino_i) {
                continue;
            }
            mino_is.push(mino_i);
            score_diff += self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        }
        let delta = weighted_delta[rnd::gen_range(0, weighted_delta.len())];
        for i in 0..r {
            let next_mino_pos = add_delta(
                self.mino_pos[mino_is[i]],
                self.mino_range[mino_is[i]],
                delta,
            );
            score_diff += self.toggle_mino(mino_is[i], next_mino_pos, true);
            next_mino_poss.push(next_mino_pos);
        }
        score_diff
    }

    fn action_move_one(
        &mut self,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
    ) -> f64 {
        let mino_i = rnd::gen_range(0, self.input.m);
        let mut score_diff = self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        let next_mino_pos = (
            rnd::gen_range(0, self.mino_range[mino_i].0),
            rnd::gen_range(0, self.mino_range[mino_i].1),
        );
        score_diff += self.toggle_mino(mino_i, next_mino_pos, true);
        mino_is.push(mino_i);
        next_mino_poss.push(next_mino_pos);
        score_diff
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
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n * input.n * 2;
    let k = input.n * 2; // :param
    let base_query_count = input.n; // :param
    let iteration = 10000;

    // 初期情報を集める
    let mut queries = investigate(k, base_query_count, &vec![], interactor, input);

    loop {
        // ミノの配置を最適化
        let mut optimizer = MinoOptimizer::new(&queries, &input);
        optimizer.optimize(iteration);

        let v = get_v(&optimizer.mino_pos, &input.minos, input.n);
        vis_v(&v, answer);

        eprintln!("mino_loss:   {:10.5}", optimizer.score);
        eprintln!("error_count: {}", error_count(&v, answer));
        eprintln!("query_count: {} / {}", interactor.query_count, query_limit);
        eprintln!("total_cost:  {:.5}", interactor.total_cost);

        let s = get_s(&v);
        if interactor.output_answer(&s) {
            exit(interactor);
        }

        // 追加情報を集める
        let query_count = base_query_count.min(query_limit - interactor.query_count - 1);
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
