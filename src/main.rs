mod def;
mod interactor;
mod param;
mod util;

use std::collections::HashSet;

use itertools::iproduct;

use crate::def::*;
use crate::interactor::*;
use crate::param::*;
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

fn create_weighted_delta_using_neighbors(max_dist: i64) -> Vec<(i64, i64)> {
    let mut delta = vec![];
    let p = 2.; // :param
    for (di, dj) in iproduct!(-max_dist..=max_dist, -max_dist..=max_dist) {
        let dist = ((i64::abs(di) + i64::abs(dj)) as f64).max(1.);
        let cnt = ((max_dist as f64 * 2.).powf(p) / dist.powf(p))
            .round()
            .max(1.) as usize;
        delta.extend(vec![(di, dj); cnt]);
    }
    delta
}

fn create_weighted_delta_using_duplication(
    max_dist: i64,
    input: &Input,
) -> Vec<Vec<Vec<(i64, i64)>>> {
    let mut weighted_delta = vec![vec![vec![]; input.m]; input.m];
    for mino_i in 0..input.m {
        for mino_j in 0..input.m {
            if mino_i == mino_j {
                continue;
            }
            let set: HashSet<&(usize, usize)> = HashSet::from_iter(input.minos[mino_j].iter());
            for d in iproduct!(-max_dist..=max_dist, -max_dist..=max_dist) {
                let mut duplicate_count = 0;
                for &p in &input.minos[mino_i] {
                    let (i, j) = (p.0 as i64 + d.0, p.1 as i64 + d.1);
                    let v = (i as usize, j as usize);
                    if set.contains(&v) {
                        duplicate_count += 1;
                    }
                }
                weighted_delta[mino_i][mino_j].extend(vec![d; duplicate_count.max(1)]);
            }
        }
    }

    weighted_delta
}

fn adjusted_q_len(x: usize) -> f64 {
    if x == 1 {
        1e-1 // :param
    } else {
        x as f64 // :param
    }
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
    fn new(
        initial_mino_pos: Option<Vec<(usize, usize)>>,
        queries: &'a Vec<(Vec<(usize, usize)>, f64)>,
        input: &'a Input,
    ) -> MinoOptimizer<'a> {
        let mino_range = get_mino_range(&input);
        let mino_pos = if let Some(initial_mino_pos) = initial_mino_pos {
            initial_mino_pos
        } else {
            let mut mino_pos = Vec::with_capacity(input.m);
            for k in 0..input.m {
                mino_pos.push((
                    rnd::gen_range(0, mino_range[k].0),
                    rnd::gen_range(0, mino_range[k].1),
                ));
            }
            mino_pos
        };

        let v = get_v(&mino_pos, &input.minos, input.n);
        let mut query_cache = vec![0.; queries.len()];
        let mut query_indices = vec![vec![vec![]; input.n]; input.n];

        let mut score = 0.;
        for (q_i, (s, x)) in queries.iter().enumerate() {
            for &(i, j) in s.iter() {
                query_cache[q_i] += v[i][j] as f64;
                query_indices[i][j].push(q_i);
            }
            score += (query_cache[q_i] - x).powf(2.) / adjusted_q_len(s.len());
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

    fn optimize(&mut self, time_limit: f64, is_anneal: bool) -> Vec<(f64, Vec<(usize, usize)>)> {
        let delta_max_dist = 2; // :param
        let delta_duplicates = create_weighted_delta_using_duplication(delta_max_dist, &self.input);
        let delta_neighbors = create_weighted_delta_using_neighbors(delta_max_dist);

        let mut mino_is = vec![];
        let mut next_mino_poss = vec![];

        // TODO: epsによっても変更する
        let start_temp = self.input.n as f64 * self.queries.len() as f64 / 1e3; // :param
        let end_temp = self.input.n as f64 * self.queries.len() as f64 / 1e5; // :param

        let mut iteration = 0;

        let mut cands = vec![];
        let mut score_log = vec![];

        let start_time = time::elapsed_seconds();

        while time::elapsed_seconds() < time_limit {
            mino_is.clear();
            next_mino_poss.clear();

            let p = rnd::nextf();
            let score_diff = if p < 0.2 {
                self.action_slide(1, &mut mino_is, &mut next_mino_poss, &delta_neighbors)
            } else if p < 0.3 {
                self.action_move_one(&mut mino_is, &mut next_mino_poss)
            } else if p < 0.9 {
                self.action_swap(2, &mut mino_is, &mut next_mino_poss, &delta_duplicates)
            } else {
                self.action_swap(3, &mut mino_is, &mut next_mino_poss, &delta_duplicates)
            };

            let adopt = if is_anneal {
                let progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
                let temp: f64 = start_temp.powf(1. - progress) * end_temp.powf(progress);
                (-score_diff / temp).exp() > rnd::nextf()
            } else {
                const EPS: f64 = 1e-6;
                score_diff < -EPS
            };
            if adopt {
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
            iteration += 1;

            if (iteration + 1) % 1000 == 0 {
                // eprintln!("[{:.4}] {:15.5}", time::elapsed_seconds(), self.score);
                cands.push((self.score, self.mino_pos.clone()));
                score_log.push(self.score);
            }
        }

        if cfg!(feature = "local") {
            use std::io::Write;
            let mut file = std::fs::File::create("score.log").unwrap();
            writeln!(&mut file, "{:?}", score_log).unwrap();
        }

        eprintln!("adopt_count: {} / {}", self.adopt_count, iteration);

        cands
    }

    fn action_swap(
        &mut self,
        r: usize,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
        weighted_delta: &Vec<Vec<Vec<(i64, i64)>>>,
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
            let weighted_delta = &weighted_delta[mino_is[i]][mino_is[(i + 1) % r]];
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
        r: usize,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
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
                let q_len = self.queries[q_i].0.len();
                score_diff -=
                    (self.query_cache[q_i] - self.queries[q_i].1).powf(2.) / adjusted_q_len(q_len);
                if turn_on {
                    self.query_cache[q_i] += 1.;
                } else {
                    self.query_cache[q_i] -= 1.;
                }
                score_diff +=
                    (self.query_cache[q_i] - self.queries[q_i].1).powf(2.) / adjusted_q_len(q_len);
            }
        }
        score_diff
    }
}

fn investigate(
    k: usize,
    query_count: usize,
    v_history: &Vec<Vec<Vec<usize>>>,
    fixed: &mut Vec<Vec<bool>>,
    interactor: &mut Interactor,
    input: &Input,
) -> Vec<(Vec<(usize, usize)>, f64)> {
    const USE_HIGH_PROB: f64 = 0.5; // :param // NOTE: 徐々に大きくした方が良さそう
    let mut queries = Vec::with_capacity(query_count);

    let mut high_prob_v = vec![];
    if v_history.len() > 0 {
        for i in 0..input.n {
            for j in 0..input.n {
                for (di, dj) in D {
                    let (ni, nj) = (i + di, j + dj);
                    if ni < input.n && nj < input.n && v_history[v_history.len() - 1][ni][nj] > 0 {
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
            if !s.contains(&a) && !fixed[a.0][a.1] {
                s.push(a);
            }
        }
        let obs_x = interactor.output_query(&s) as f64;
        let obs_x = if k > 1 {
            ((obs_x - k as f64 * input.eps) / (1. - 2. * input.eps)).max(0.) // NOTE: 本当にあってる？
        } else {
            obs_x
        };
        if s.len() == 1 {
            fixed[s[0].0][s[0].1] = true;
        }
        queries.push((s, obs_x));
    }

    queries
}

#[allow(unused)]
fn solve_data_collection(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n.pow(2) * 2;
    let k = input.n * 2; // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut checked_s = HashSet::new();

    let mut query_count = 10;

    while query_count < query_limit {
        query_count += 10;
        let mut _queries = investigate(
            k,
            query_count.min(query_limit - 1),
            &vec![],
            &mut fixed,
            interactor,
            input,
        );
        queries.extend(_queries);

        // ミノの配置を最適化
        let mut optimizer = MinoOptimizer::new(None, &queries, &input);
        let mut cands = optimizer.optimize(time::elapsed_seconds() + 1.0, true);
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (mino_loss, mino_pos) in cands.iter().take(5) {
            let v = get_v(&mino_pos, &input.minos, input.n);
            let s = get_s(&v);
            if checked_s.contains(&s) {
                continue;
            }

            vis_v(&v, answer);
            eprint!("mino_loss:   {:10.5}", mino_loss);
            eprintln!(", error_count: {}", error_count(&v, answer));
            // eprintln!("query_count: {} / {}", interactor.query_count, query_limit);
            // eprintln!("total_cost:  {:.5}", interactor.total_cost);

            if error_count(&v, answer) == 0 {
                if interactor.output_answer(&s) {
                    exit(interactor);
                }
            }

            checked_s.insert(s);
        }
    }
}

#[allow(unused)]
fn solve_greedy(interactor: &mut Interactor, input: &Input) {
    let mut s = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let x = interactor.output_query(&vec![(i, j)]);
            if x > 0 {
                s.push((i, j));
            }
        }
    }
    assert!(interactor.output_answer(&s));
    exit(interactor);
}

fn get_query_count(input: &Input) -> usize {
    let query_limit = input.n.pow(2) * 2;
    let n = input.n as f64;
    let m = input.m as f64;
    let eps = input.eps;
    let dense = input.minos.iter().map(|mino| mino.len()).sum::<usize>() as f64 / (n * n);

    let pred = query_count_linear_regression(n, m, eps, dense);
    let pred = pred * query_limit as f64;

    pred.round() as usize
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let time_limit = 2.8;
    let query_limit = input.n.pow(2) * 2;
    let k = (input.n as f64 * (5. - input.eps * 20.)).round() as usize; // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut checked_s = HashSet::new();
    let mut v_history = vec![];

    let base_query_count = get_query_count(input).clamp(20, query_limit);
    eprintln!("base_query_count = {}", base_query_count);

    let steps: Vec<f64> = vec![0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
        .into_iter()
        .filter(|x| x * (base_query_count as f64) < query_limit as f64)
        .collect();
    let step_sum: f64 = steps.iter().map(|x| x.sqrt()).sum();
    let step_ratio: Vec<f64> = steps.iter().map(|x| x.sqrt() / step_sum).collect();

    for i in 1..steps.len() {
        let query_count = if i < steps.len() - 1 {
            (((steps[i] - steps[i - 1]) * base_query_count as f64).round() as usize)
                .clamp(0, query_limit - interactor.query_count - i)
        } else {
            (query_limit as i64 - interactor.query_count as i64 - 5).max(0) as usize
        };
        let mut _queries = investigate(k, query_count, &v_history, &mut fixed, interactor, input);
        queries.extend(_queries);

        // ミノの配置を最適化
        let optimize_time_limit = if i < steps.len() - 1 {
            time_limit * step_ratio[i]
        } else {
            time_limit - time::elapsed_seconds()
        };
        let mut optimizer = MinoOptimizer::new(None, &queries, &input);
        let mut cands = optimizer.optimize(time::elapsed_seconds() + optimize_time_limit, true);
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut tested_count = 0;

        for (mino_loss, mino_pos) in cands.iter() {
            if tested_count >= i {
                break;
            }
            let v = get_v(&mino_pos, &input.minos, input.n);
            let s = get_s(&v);
            if checked_s.contains(&s) {
                continue;
            }

            // vis_v(&v, answer);
            eprintln!("mino_loss:   {:10.5}", mino_loss);
            eprintln!("error_count: {}", error_count(&v, answer));
            eprintln!("query_count: {} / {}", interactor.query_count, query_limit);
            eprintln!("total_cost:  {:.5}", interactor.total_cost);

            if interactor.output_answer(&s) {
                // vis_queries(&queries, &input);
                exit(interactor);
            }

            checked_s.insert(s);
            v_history.push(v);
            tested_count += 1;
        }
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

    // クエリを最後まで消費する
    loop {
        interactor.output_answer(&vec![(0, 0)]);
    }
}
