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
    delta_duplicates: Vec<Vec<(i64, i64)>>,
    mino_range: Vec<(usize, usize)>,
}

struct MinoOptimizer {
    input_util: InputUtility,
    query_cache: Vec<f64>,
    query_indices: Vec<Vec<usize>>,
    mino_pos: Vec<(usize, usize)>,
    score: f64,
    adopt_count: usize,
    queries: Vec<(Vec<(usize, usize)>, f64)>,
    inv_query_size: f64,
    input: Input,
}

impl MinoOptimizer {
    fn new(param: &Param, input: &Input) -> MinoOptimizer {
        let input_util = InputUtility {
            delta_duplicates: get_weighted_delta_using_duplication(
                param.delta_max_dist,
                param.duplicate_min_d,
                &input,
            ),
            delta_neighbors: get_weighted_delta_using_neighbors(
                param.delta_max_dist,
                param.neighbor_p,
            ),
            mino_range: get_mino_range(&input),
        };
        let mino_pos = {
            let mut mino_pos = Vec::with_capacity(input.m);
            for k in 0..input.m {
                mino_pos.push((
                    rnd::gen_index(input_util.mino_range[k].0),
                    rnd::gen_index(input_util.mino_range[k].1),
                ));
            }
            mino_pos
        };

        MinoOptimizer {
            input_util,
            query_cache: vec![],
            query_indices: vec![vec![]; input.n * input.n],
            mino_pos,
            score: 0.,
            adopt_count: 0,
            queries: vec![],
            inv_query_size: 1. / param.get_query_size(input) as f64,
            input: input.clone(),
        }
    }

    fn add_queries(&mut self, queries: Vec<(Vec<(usize, usize)>, f64)>) {
        let v = get_v(&self.mino_pos, &self.input.minos, self.input.n);
        self.query_cache.reserve(queries.len());
        for (i, (s, x)) in queries.iter().enumerate() {
            let q_i = self.queries.len() + i;
            let mut v_sum = 0;
            for &(i, j) in s.iter() {
                v_sum += v[i][j];
                self.query_indices[i * self.input.n + j].push(q_i);
            }
            let v_sum = v_sum as f64;
            self.query_cache.push(v_sum);
            self.score += calc_error(*x, v_sum, self.inv_query_size);
        }
        self.queries.extend(queries);
    }

    fn optimize(&mut self, param: &Param, time_limit: f64) -> Vec<(f64, Vec<(usize, usize)>)> {
        let mut mino_is = Vec::with_capacity(3);
        let mut next_mino_poss = Vec::with_capacity(3);

        let action_ratio = param.get_action_ratio();
        let b = (self.input.n * self.queries.len()) as f64;
        let start_temp = b / param.start_temp_coef;
        let end_temp = b / param.end_temp_coef;

        let mut iteration = 0;

        let mut cands = vec![];
        let start_time = time::elapsed_seconds();

        while time::elapsed_seconds() < time_limit {
            mino_is.clear();
            next_mino_poss.clear();

            let query_cache_copy = self.query_cache.clone();

            let p = rnd::nextf();
            if p < action_ratio[0] {
                self.action_slide(1, &mut mino_is, &mut next_mino_poss)
            } else if p < action_ratio[1] {
                self.action_move_one(&mut mino_is, &mut next_mino_poss)
            } else if p < action_ratio[2] {
                self.action_swap(2, &mut mino_is, &mut next_mino_poss)
            } else {
                self.action_swap(3, &mut mino_is, &mut next_mino_poss)
            };

            let progress = (time::elapsed_seconds() - start_time) / (time_limit - start_time);
            let temp: f64 = start_temp.powf(1. - progress) * end_temp.powf(progress);
            let threshold = self.score - temp * rnd::nextf().max(1e-6).ln();
            let mut new_score = 0.;
            for (i, (_, x)) in self.queries.iter().enumerate() {
                if new_score > threshold {
                    break;
                }
                new_score += calc_error(*x, self.query_cache[i], self.inv_query_size);
            }
            let adopt = new_score <= threshold;

            if adopt {
                self.score = new_score;
                for i in 0..mino_is.len() {
                    self.mino_pos[mino_is[i]] = next_mino_poss[i];
                }
                cands.push((self.score, self.mino_pos.clone()));
                self.adopt_count += 1;
            } else {
                self.query_cache = query_cache_copy;
            }
            iteration += 1;
        }

        eprintln!("adopt_count: {} / {}", self.adopt_count, iteration);
        cands
    }

    fn action_swap(
        &mut self,
        r: usize,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
    ) {
        let r = r.min(self.mino_pos.len());
        while mino_is.len() < r {
            let mino_i = rnd::gen_index(self.mino_pos.len());
            if mino_is.contains(&mino_i) {
                continue;
            }
            mino_is.push(mino_i);
            self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        }

        for i in 0..r {
            let weighted_delta =
                &self.input_util.delta_duplicates[mino_is[i] * self.input.m + mino_is[(i + 1) % r]];
            let delta = weighted_delta[rnd::gen_index(weighted_delta.len())];
            let next_mino_pos = add_delta(
                self.mino_pos[mino_is[(i + 1) % r]],
                self.input_util.mino_range[mino_is[i]],
                delta,
            );
            self.toggle_mino(mino_is[i], next_mino_pos, true);
            next_mino_poss.push(next_mino_pos);
        }
    }

    fn action_slide(
        &mut self,
        r: usize,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
    ) {
        let r = r.min(self.mino_pos.len());
        while mino_is.len() < r {
            let mino_i = rnd::gen_index(self.mino_pos.len());
            if mino_is.contains(&mino_i) {
                continue;
            }
            mino_is.push(mino_i);
            self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        }
        for i in 0..r {
            let delta = self.input_util.delta_neighbors
                [rnd::gen_index(self.input_util.delta_neighbors.len())];
            let next_mino_pos = add_delta(
                self.mino_pos[mino_is[i]],
                self.input_util.mino_range[mino_is[i]],
                delta,
            );
            self.toggle_mino(mino_is[i], next_mino_pos, true);
            next_mino_poss.push(next_mino_pos);
        }
    }

    fn action_move_one(
        &mut self,
        mino_is: &mut Vec<usize>,
        next_mino_poss: &mut Vec<(usize, usize)>,
    ) {
        let mino_i = rnd::gen_index(self.mino_pos.len());
        self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
        let next_mino_pos = (
            rnd::gen_index(self.input_util.mino_range[mino_i].0),
            rnd::gen_index(self.input_util.mino_range[mino_i].1),
        );
        self.toggle_mino(mino_i, next_mino_pos, true);
        mino_is.push(mino_i);
        next_mino_poss.push(next_mino_pos);
    }

    fn toggle_mino(&mut self, mino_i: usize, mino_pos: (usize, usize), turn_on: bool) {
        let d = if turn_on { 1. } else { -1. };
        for &(_i, _j) in self.input.minos[mino_i].iter() {
            let (i, j) = (mino_pos.0 + _i, mino_pos.1 + _j);
            for &q_i in self.query_indices[i * self.input.n + j].iter() {
                self.query_cache[q_i] += d;
            }
        }
    }
}

fn calc_v_prob(
    top_k: usize,
    cands: &Vec<(f64, Vec<Vec<usize>>)>,
    param: &Param,
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
            var[i][j] += mean[i][j] * param.v_prob_mean_w;
            var[i][j] = var[i][j].max(param.v_prob_min_var);
            var_sum += var[i][j];
        }
    }
    let alpha = 10000. / var_sum;
    let mut prob_v = Vec::with_capacity(11000);
    for i in 0..input.n {
        for j in 0..input.n {
            let cnt = (var[i][j] * alpha).round().max(1.) as usize;
            prob_v.extend(vec![(i, j); cnt]);
        }
    }

    prob_v
}

fn create_query(
    query_size: usize,
    prob_v: &Vec<(usize, usize)>,
    param: &Param,
    input: &Input,
) -> Vec<(usize, usize)> {
    let mut s = Vec::with_capacity(query_size);
    let p = param.p_max - (input.m as f64).powf(2.) / 400. * (param.p_max - param.p_min);
    while s.len() < query_size {
        let a = prob_v[rnd::gen_index(prob_v.len())];
        if s.contains(&a) {
            continue;
        }
        s.push(a);
        while s.len() < query_size && rnd::nextf() < p {
            let d = D[rnd::gen_index(D.len())];
            let b = (a.0 + d.0, a.1 + d.1);
            if b.0 < input.n && b.1 < input.n && !s.contains(&b) {
                s.push(b);
            }
        }
    }
    s
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
            continue;
        }

        if interactor.output_answer(&s) {
            eprintln!("mino_loss:   {:10.5}", err);
            eprintln!("error_count: {}", error_count(&v, answer));
            eprintln!("query_count: {}", interactor.query_count);
            eprintln!("total_cost:  {:.5}", interactor.total_cost);
            exit(interactor);
        } else {
            checked_s.insert(s);
        }
    }
}

fn solve(interactor: &mut Interactor, input: &Input, param: &Param, answer: &Option<Answer>) {
    const OUT_LIM: usize = 5;
    let time_limit = if cfg!(feature = "local") { 2.0 } else { 2.9 };
    let query_limit = input.n.pow(2) * 2;
    let top_k = 100;
    let query_size = param.get_query_size(input);
    eprintln!("query_size: {}", query_size);

    let mut queries = vec![];
    let mut checked_s = HashSet::new();
    let mut cands: Vec<(f64, Vec<Vec<usize>>)> = vec![];
    let mut answer_set: HashSet<Vec<(usize, usize)>> = HashSet::new();

    let base_query_count = param
        .get_query_count(input)
        .clamp(10, query_limit - OUT_LIM);
    eprintln!("base_query_count = {}", base_query_count);

    let steps = param.get_steps();
    let steps: Vec<f64> = steps
        .into_iter()
        .filter(|x| x * (base_query_count as f64) < query_limit as f64)
        .collect();
    let step_sum: f64 = steps.iter().map(|&x| x.powf(param.step_p)).sum();
    let step_ratio: Vec<f64> = steps
        .iter()
        .map(|&x| x.powf(param.step_p) / step_sum)
        .collect();
    let steps: Vec<usize> = steps
        .into_iter()
        .map(|x| (x * base_query_count as f64).round() as usize)
        .collect();
    eprintln!("steps: {:?}", steps);
    let mut next_step = 0;

    let mut optimizer = MinoOptimizer::new(param, input);
    let mut prob_v = calc_v_prob(top_k, &cands, param, input);

    while time::elapsed_seconds() < 2.98 && interactor.query_count + 1 < query_limit {
        // 調査
        let s = create_query(query_size, &prob_v, param, input);
        let obs_x = (interactor.output_query(&s) as f64 - query_size as f64 * input.eps)
            / (1. - 2. * input.eps);
        for (err, v) in cands.iter_mut() {
            let mut v_sum = 0.;
            for &(i, j) in s.iter() {
                v_sum += v[i][j] as f64;
            }
            *err += calc_error(v_sum, obs_x, optimizer.inv_query_size);
        }
        queries.push((s, obs_x));

        if next_step >= steps.len() || interactor.query_count < steps[next_step] {
            continue;
        }

        let is_final_step = next_step == steps.len() - 1;

        // 最適化
        let optimize_time_limit = if is_final_step {
            time_limit - time::elapsed_seconds()
        } else {
            time_limit * step_ratio[next_step]
        };
        optimizer.add_queries(queries);
        queries = vec![];

        let mut new_cands =
            optimizer.optimize(param, time::elapsed_seconds() + optimize_time_limit);
        new_cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 候補の追加
        for (err, mino_pos) in new_cands.into_iter() {
            if cands.len() > top_k * 2 || (!cands.is_empty() && err > cands[0].0 * 2.) {
                break;
            }
            if answer_set.contains(&mino_pos) {
                continue;
            }
            answer_set.insert(mino_pos.clone());
            let v = get_v(&mino_pos, &input.minos, input.n);
            cands.push((err, v));
        }

        let out_cnt = if is_final_step {
            1000
        } else {
            ((interactor.query_count as f64 / param.out_step_size).ceil() as usize).min(OUT_LIM)
        };
        output_answer(
            out_cnt,
            top_k,
            &mut cands,
            interactor,
            &mut checked_s,
            answer,
        );

        next_step += 1;
        prob_v = calc_v_prob(top_k, &cands, param, input);
    }

    output_answer(1000, top_k, &mut cands, interactor, &mut checked_s, answer);
}

struct Param {
    min_k: f64,
    max_k: f64,
    k_p: f64,
    start_step: f64,
    end_step: f64,
    step_cnt: usize,
    v_prob_mean_w: f64,
    v_prob_min_var: f64,
    p_max: f64,
    p_min: f64,
    delta_max_dist: i64,
    neighbor_p: f64,
    duplicate_min_d: usize,
    step_p: f64,
    out_step_size: f64,
    start_temp_coef: f64,
    end_temp_coef: f64,
    action_slide_ratio: f64,
    action_move_one: f64,
    action_swap_2: f64,
    action_swap_3: f64,
}

impl Param {
    fn get_steps(&self) -> Vec<f64> {
        let mut steps = Vec::with_capacity(self.step_cnt);
        let step_width = (self.end_step - self.start_step) / (self.step_cnt - 1) as f64;
        for i in 0..self.step_cnt {
            steps.push(self.start_step + i as f64 * step_width);
        }
        steps
    }

    fn get_query_count(&self, input: &Input) -> usize {
        let n = input.n as f64;
        let m = input.m as f64;
        let eps = input.eps;
        let dense = input.minos.iter().map(|mino| mino.len()).sum::<usize>() as f64 / (n * n);

        let pred = query_count_linear_regression(n, m, eps, dense);

        pred.round().max(1.) as usize
    }

    fn get_query_size(&self, input: &Input) -> usize {
        const EPS_MAX: f64 = 0.2;
        let a = (self.max_k - self.min_k) / EPS_MAX.powf(self.k_p);
        (input.n as f64 * (self.max_k - input.eps.powf(self.k_p) * a)).round() as usize
    }

    fn get_action_ratio(&self) -> Vec<f64> {
        let mut p = vec![
            self.action_slide_ratio,
            self.action_move_one,
            self.action_swap_2,
            self.action_swap_3,
        ];
        let p_sum: f64 = p.iter().sum();
        for e in p.iter_mut() {
            *e /= p_sum;
        }
        for i in 0..p.len() - 1 {
            p[i + 1] += p[i];
        }
        p
    }
}

fn load_params() -> Param {
    let load_from_cmd_args = false;
    if load_from_cmd_args {
        use std::env;
        let args: Vec<String> = env::args().collect();
        Param {
            min_k: args[1].parse().unwrap(),
            max_k: args[2].parse().unwrap(),
            k_p: args[3].parse().unwrap(),
            start_step: args[4].parse().unwrap(),
            end_step: args[5].parse().unwrap(),
            step_cnt: args[6].parse().unwrap(),
            v_prob_mean_w: args[7].parse().unwrap(),
            v_prob_min_var: args[8].parse().unwrap(),
            p_max: args[9].parse().unwrap(),
            p_min: args[10].parse().unwrap(),
            delta_max_dist: args[11].parse().unwrap(),
            neighbor_p: args[12].parse().unwrap(),
            duplicate_min_d: args[13].parse().unwrap(),
            step_p: args[14].parse().unwrap(),
            out_step_size: args[15].parse().unwrap(),
            start_temp_coef: args[16].parse().unwrap(),
            end_temp_coef: args[17].parse().unwrap(),
            action_slide_ratio: args[18].parse().unwrap(),
            action_move_one: args[19].parse().unwrap(),
            action_swap_2: args[20].parse().unwrap(),
            action_swap_3: args[21].parse().unwrap(),
        }
    } else {
        Param {
            min_k: 3.,
            max_k: 5.5,
            k_p: 0.85,
            start_step: 0.72,
            end_step: 1.8,
            step_cnt: 5,
            v_prob_mean_w: 1.460,
            v_prob_min_var: 0.027,
            p_max: 0.989,
            p_min: 0.421,
            delta_max_dist: 2,
            neighbor_p: 2.,
            duplicate_min_d: 1,
            step_p: 1.,
            out_step_size: 100.,
            start_temp_coef: 1e3,
            end_temp_coef: 1e5,
            action_slide_ratio: 0.2,
            action_move_one: 0.1,
            action_swap_2: 0.6,
            action_swap_3: 0.1,
        }
    }
}

fn main() {
    time::start_clock();
    let mut interactor = Interactor::new();
    let input = interactor.read_input();
    let param = load_params();
    let answer = if cfg!(feature = "local") {
        Some(interactor.read_answer(&input))
    } else {
        None
    };

    solve(&mut interactor, &input, &param, &answer);

    // クエリを最後まで消費する
    loop {
        interactor.output_answer(&vec![(0, 0)]);
    }
}
