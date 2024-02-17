mod def;
mod interactor;
mod param;
mod util;

use crate::def::*;
use crate::interactor::*;
use crate::param::*;
use crate::util::*;
use std::collections::HashSet;

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

struct InputUtility {
    delta_neighbors: Vec<(i64, i64)>,
    delta_duplicates: Vec<Vec<Vec<(i64, i64)>>>,
    mino_range: Vec<(usize, usize)>,
}

struct MinoOptimizer {
    input_util: InputUtility,
    query_cache: Vec<f64>,
    query_indices: Vec<Vec<Vec<usize>>>,
    mino_pos: Vec<(usize, usize)>,
    score: f64,
    adopt_count: usize,
    queries: Vec<(Vec<(usize, usize)>, f64)>,
    input: Input,
}

impl MinoOptimizer {
    fn new(input: &Input) -> MinoOptimizer {
        let delta_max_dist = 2; // :param
        let input_util = InputUtility {
            delta_duplicates: get_weighted_delta_using_duplication(delta_max_dist, &input),
            delta_neighbors: get_weighted_delta_using_neighbors(delta_max_dist),
            mino_range: get_mino_range(&input),
        };
        let mino_pos = {
            let mut mino_pos = Vec::with_capacity(input.m);
            for k in 0..input.m {
                mino_pos.push((
                    rnd::gen_range(0, input_util.mino_range[k].0),
                    rnd::gen_range(0, input_util.mino_range[k].1),
                ));
            }
            mino_pos
        };

        MinoOptimizer {
            input_util,
            query_cache: vec![],
            query_indices: vec![vec![vec![]; input.n]; input.n],
            mino_pos,
            score: 0.,
            adopt_count: 0,
            queries: vec![],
            input: input.clone(),
        }
    }

    fn add_queries(&mut self, queries: Vec<(Vec<(usize, usize)>, f64)>) {
        let v = get_v(&self.mino_pos, &self.input.minos, self.input.n);
        self.query_cache.reserve(queries.len());
        for (i, (s, x)) in queries.iter().enumerate() {
            let q_i = self.queries.len() + i;
            let mut v_sum = 0.;
            for &(i, j) in s.iter() {
                v_sum += v[i][j] as f64;
                self.query_indices[i][j].push(q_i);
            }
            self.query_cache.push(v_sum);
            self.score += calc_error(*x, v_sum, s.len());
        }
        self.queries.extend(queries);
    }

    fn optimize(&mut self, time_limit: f64, is_anneal: bool) -> Vec<(f64, Vec<(usize, usize)>)> {
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
                self.action_slide(1, &mut mino_is, &mut next_mino_poss)
            } else if p < 0.3 {
                self.action_move_one(&mut mino_is, &mut next_mino_poss)
            } else if p < 0.9 {
                self.action_swap(2, &mut mino_is, &mut next_mino_poss)
            } else {
                self.action_swap(3, &mut mino_is, &mut next_mino_poss)
            };

            let adopt = if is_anneal {
                let progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
                let temp: f64 = start_temp.powf(1. - progress) * end_temp.powf(progress);
                // let temp = start_temp * (1. - progress) + end_temp * progress;
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
                cands.push((self.score, self.mino_pos.clone()));
                self.adopt_count += 1;
            } else {
                for i in 0..mino_is.len() {
                    self.toggle_mino(mino_is[i], next_mino_poss[i], false);
                    self.toggle_mino(mino_is[i], self.mino_pos[mino_is[i]], true);
                }
            }
            iteration += 1;

            if (iteration + 1) % 100 == 0 {
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
    ) -> f64 {
        let r = r.min(self.mino_pos.len());
        let mut score_diff = 0.;
        while mino_is.len() < r {
            let mino_i = rnd::gen_range(0, self.mino_pos.len());
            if mino_is.contains(&mino_i) {
                continue;
            }
            mino_is.push(mino_i);
            score_diff += self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        }

        for i in 0..r {
            let weighted_delta =
                &self.input_util.delta_duplicates[mino_is[i]][mino_is[(i + 1) % r]];
            let delta = weighted_delta[rnd::gen_range(0, weighted_delta.len())];
            let next_mino_pos = add_delta(
                self.mino_pos[mino_is[(i + 1) % r]],
                self.input_util.mino_range[mino_is[i]],
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
    ) -> f64 {
        let r = r.min(self.mino_pos.len());
        let mut score_diff = 0.;
        while mino_is.len() < r {
            let mino_i = rnd::gen_range(0, self.mino_pos.len());
            if mino_is.contains(&mino_i) {
                continue;
            }
            mino_is.push(mino_i);
            score_diff += self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        }
        let delta = self.input_util.delta_neighbors
            [rnd::gen_range(0, self.input_util.delta_neighbors.len())];
        for i in 0..r {
            let next_mino_pos = add_delta(
                self.mino_pos[mino_is[i]],
                self.input_util.mino_range[mino_is[i]],
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
        let mino_i = rnd::gen_range(0, self.mino_pos.len());
        let mut score_diff = self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        let next_mino_pos = (
            rnd::gen_range(0, self.input_util.mino_range[mino_i].0),
            rnd::gen_range(0, self.input_util.mino_range[mino_i].1),
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
                score_diff -= calc_error(self.query_cache[q_i], self.queries[q_i].1, q_len);
                if turn_on {
                    self.query_cache[q_i] += 1.;
                } else {
                    self.query_cache[q_i] -= 1.;
                }
                score_diff += calc_error(self.query_cache[q_i], self.queries[q_i].1, q_len);
            }
        }
        score_diff
    }
}

fn get_query_count(input: &Input) -> usize {
    let n = input.n as f64;
    let m = input.m as f64;
    let eps = input.eps;
    let dense = input.minos.iter().map(|mino| mino.len()).sum::<usize>() as f64 / (n * n);

    let pred = query_count_linear_regression(n, m, eps, dense);

    // let query_limit = input.n.pow(2) * 2;
    // let pred = pred * query_limit as f64;

    pred.round().max(1.) as usize
}

fn get_query_size(input: &Input) -> usize {
    (input.n as f64 * (5. - input.eps * 20.)).round() as usize
}

fn calc_high_prob(
    top_k: usize,
    cands: &Vec<(f64, Vec<Vec<usize>>)>,
    input: &Input,
) -> Vec<(usize, usize)> {
    let mut mean = vec![vec![0.; input.n]; input.n];
    let top_k = top_k.min(cands.len());
    for i in 0..input.n {
        for j in 0..input.n {
            for (_, v) in cands.iter().take(top_k) {
                for (di, dj) in D {
                    let (ni, nj) = (i + di, j + dj);
                    if ni < input.n && nj < input.n && v[ni][nj] > 0 {
                        mean[i][j] += 1.;
                        break;
                    }
                }
            }
            mean[i][j] /= top_k as f64;
        }
    }
    let mut var = vec![vec![0.; input.n]; input.n];
    let mut var_sum = 0.;
    for i in 0..input.n {
        for j in 0..input.n {
            for (_, v) in cands.iter().take(top_k) {
                var[i][j] += (v[i][j] as f64 - mean[i][j]).powf(2.);
            }
            var[i][j] /= top_k as f64;
            var[i][j] += mean[i][j];
            var[i][j] = var[i][j].max(0.1);
            var_sum += var[i][j];
        }
    }
    let alpha = 10000. / var_sum;
    let mut prob_v = Vec::with_capacity(11000);
    for i in 0..input.n {
        for j in 0..input.n {
            let cnt = (var[i][j] * alpha).round().max(1.) as usize;
            prob_v.extend(vec![(i, j); cnt]);
            // eprint!("{:8.5} ", var[i][j]);
        }
        // eprintln!();
    }

    prob_v
}

fn output_answer(
    out_cnt: usize,
    top_k: usize,
    cands: &mut Vec<(f64, Vec<Vec<usize>>)>,
    interactor: &mut Interactor,
    checked_s: &mut HashSet<Vec<(usize, usize)>>,
    answer: &Option<Answer>,
) {
    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    cands.truncate(top_k);
    for (err, v) in cands.iter_mut().take(out_cnt) {
        let s = get_s(&v);
        if checked_s.contains(&s) {
            *err = 1e50;
            continue;
        }
        eprintln!("mino_loss:   {:10.5}", err);
        eprintln!("error_count: {}", error_count(&v, answer));
        eprintln!("query_count: {}", interactor.query_count);
        eprintln!("total_cost:  {:.5}", interactor.total_cost);

        if interactor.output_answer(&s) {
            exit(interactor);
        } else {
            *err = 1e50;
            checked_s.insert(s);
        }
    }
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let time_limit = 2.8;
    let query_limit = input.n.pow(2) * 2;
    let top_k = 100;
    let query_size = get_query_size(input); // :param

    let mut queries = vec![];
    let fixed = vec![vec![false; input.n]; input.n];
    let mut checked_s = HashSet::new();
    let mut cands: Vec<(f64, Vec<Vec<usize>>)> = vec![];
    let mut answer_set: HashSet<Vec<Vec<usize>>> = HashSet::new();

    let base_query_count = get_query_count(input).clamp(10, query_limit);
    eprintln!("base_query_count = {}", base_query_count);

    let steps = vec![0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]; // TODO: base_query_countごとに調整する
    let steps: Vec<usize> = steps
        .into_iter()
        .filter(|x| x * (base_query_count as f64) < query_limit as f64)
        .map(|x| (x * base_query_count as f64).round() as usize)
        .collect();
    let step_sum: usize = steps.iter().map(|&x| x as usize).sum();
    let step_ratio: Vec<f64> = steps.iter().map(|&x| x as f64 / step_sum as f64).collect();
    let mut next_step = 0;

    let mut optimizer = MinoOptimizer::new(&input);

    loop {
        if next_step >= steps.len() {
            break;
        }

        let prob_v = calc_high_prob(top_k, &cands, input);

        // 調査
        let mut s = vec![];
        while s.len() < query_size {
            let a = prob_v[rnd::gen_range(0, prob_v.len())];
            if !s.contains(&a) && !fixed[a.0][a.1] {
                s.push(a);
            }
        }
        let obs_x = interactor.output_query(&s) as f64;
        let obs_x = ((obs_x - query_size as f64 * input.eps) / (1. - 2. * input.eps)).max(0.);

        for (err, v) in cands.iter_mut() {
            let mut v_sum = 0.;
            for &(i, j) in s.iter() {
                v_sum += v[i][j] as f64;
            }
            *err += calc_error(v_sum, obs_x, s.len());
        }
        queries.push((s, obs_x));

        if interactor.query_count >= steps[next_step] {
            // 最適化
            let optimize_time_limit = if next_step == steps.len() {
                time_limit - time::elapsed_seconds()
            } else {
                time_limit * step_ratio[next_step]
            };

            optimizer.add_queries(queries);
            queries = vec![];

            let mut new_cands =
                optimizer.optimize(time::elapsed_seconds() + optimize_time_limit, true);
            new_cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // 候補の追加
            for (err, mino_pos) in new_cands.into_iter() {
                if cands.len() > top_k * 2 || (!cands.is_empty() && err > cands[0].0 * 2.) {
                    break;
                }
                let v = get_v(&mino_pos, &input.minos, input.n);
                if answer_set.contains(&v) {
                    continue;
                }
                answer_set.insert(v.clone());
                cands.push((err, v));
            }

            output_answer(
                next_step,
                top_k,
                &mut cands,
                interactor,
                &mut checked_s,
                answer,
            );

            next_step += 1;
        }
    }

    // 最後まで出力する
    output_answer(1000, top_k, &mut cands, interactor, &mut checked_s, answer);
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
