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

struct MinoOptimizer<'a> {
    input_util: InputUtility,
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
        let delta_max_dist = 2; // :param
        let input_util = InputUtility {
            delta_duplicates: get_weighted_delta_using_duplication(delta_max_dist, &input),
            delta_neighbors: get_weighted_delta_using_neighbors(delta_max_dist),
            mino_range: get_mino_range(&input),
        };
        let mino_pos = if let Some(initial_mino_pos) = initial_mino_pos {
            initial_mino_pos
        } else {
            let mut mino_pos = Vec::with_capacity(input.m);
            for k in 0..input.m {
                mino_pos.push((
                    rnd::gen_range(0, input_util.mino_range[k].0),
                    rnd::gen_range(0, input_util.mino_range[k].1),
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
            score += calc_error(query_cache[q_i], *x, s.len());
        }

        MinoOptimizer {
            input_util,
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
        let mut mino_is = vec![];
        let mut next_mino_poss = vec![];

        // TODO: epsによっても変更する
        let start_temp = self.input.n as f64 * self.queries.len() as f64 / 1e3; // :param
        let end_temp = self.input.n as f64 * self.queries.len() as f64 / 1e5; // :param

        let mut iteration = 0;

        let mut cands = vec![];
        let mut score_log = vec![];

        let start_time = time::elapsed_seconds();
        let mut best_score = self.score;

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

            if (iteration + 1) % 100 == 0 || self.score < best_score {
                score_log.push(self.score);
            }
            best_score = best_score.min(self.score);
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
        let mino_i = rnd::gen_range(0, self.input.m);
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
fn solve_answer_contains(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n.pow(2) * 2;
    let query_size = get_query_size(input); // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut v_history = vec![];

    let base_query_count = get_query_count(input).clamp(10, query_limit);
    eprintln!("base_query_count = {}", base_query_count);

    let mut _queries = investigate(
        query_size,
        (base_query_count / 4)
            .min(query_limit - interactor.query_count - 1)
            .max(1),
        &vec![],
        &mut fixed,
        interactor,
        input,
    );
    queries.extend(_queries);

    let mut optimizer = MinoOptimizer::new(None, &queries, &input);
    let mut cands = optimizer.optimize(time::elapsed_seconds() + 1.0, true);
    cands.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (_, mino_pos) = &cands[0];
    let v = get_v(&mino_pos, &input.minos, input.n);
    v_history.push(v);

    let mut _queries = investigate(
        query_size,
        (base_query_count / 2)
            .min(query_limit - interactor.query_count - 1)
            .max(1),
        &v_history,
        &mut fixed,
        interactor,
        input,
    );
    queries.extend(_queries);

    for _ in 0..1 {
        let mut sampled_queries = &queries;
        // let mut sampled_queries = vec![];
        // for _ in 0..queries.len() / 2 {
        //     sampled_queries.push(queries[rnd::gen_range(0, queries.len())].clone());
        // }

        let mut optimizer = MinoOptimizer::new(None, &sampled_queries, &input);
        let mut cands = optimizer.optimize(time::elapsed_seconds() + 2.0, true);
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

        eprintln!(
            "query_count: {}, cand sizes: {}",
            interactor.query_count,
            cands.len()
        );

        let mut checked_v = HashSet::new();
        for (mino_loss, mino_pos) in cands.iter() {
            let v = get_v(&mino_pos, &input.minos, input.n);
            let mut s = get_s(&v);
            s.sort();
            if checked_v.contains(&v) {
                continue;
            }
            if let Some(answer) = answer {
                let mut ans_s = get_s(&answer.v);
                ans_s.sort();
                if ans_s == s {
                    eprintln!("---------------- contained!!!!!! -----------------: ");
                    eprintln!(
                        "cand_i: {}, query_count: {}",
                        checked_v.len(),
                        interactor.query_count
                    );
                    eprintln!("answer_mino_loss: {:.5}", mino_loss);
                }

                if checked_v.len() == 0 {
                    eprintln!("mino_loss: {:.5}", mino_loss);
                    let mut cnt = 0;
                    for i in 0..input.m {
                        if mino_pos[i] != answer.mino_pos[i] {
                            cnt += 1;
                        }
                    }
                    eprintln!("miss_count: {}", cnt);
                }
            }

            if error_count(&v, answer) == 0 || checked_v.len() <= 5 {
                if interactor.output_answer(&s) {
                    exit(interactor);
                }
            }

            checked_v.insert(v);
        }

        if let Some(answer) = answer {
            let v = &answer.v;
            let mut score = 0.;
            for (s, x) in optimizer.queries.iter() {
                let mut v_sum = 0.;
                for &(i, j) in s.iter() {
                    v_sum += v[i][j] as f64;
                }
                score += calc_error(v_sum, *x, s.len());
            }
            eprintln!("ans_loss: {:.5}", score);
        }
        eprintln!("checked_v: {}", checked_v.len());
    }
}

#[allow(unused)]
fn solve_data_collection(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n.pow(2) * 2;
    let query_size = get_query_size(input); // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut checked_s = HashSet::new();

    let mut query_count = 6;

    while interactor.query_count < query_limit {
        let next_query_count = ((query_count as f64 * 0.1).ceil() as usize).max(2);
        let mut _queries = investigate(
            query_size,
            next_query_count
                .min(query_limit - interactor.query_count - 1)
                .max(1),
            &vec![],
            &mut fixed,
            interactor,
            input,
        );
        query_count += next_query_count;
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

            // vis_v(&v, answer);
            // eprint!("mino_loss:   {:10.5}", mino_loss);
            // eprintln!(", error_count: {}", error_count(&v, answer));
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

            let mut optimizer = MinoOptimizer::new(None, &queries, &input);
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

            next_step += 1;

            // 答える
            output_answer(
                next_step,
                top_k,
                &mut cands,
                interactor,
                &mut checked_s,
                answer,
            );
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
