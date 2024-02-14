mod def;
mod interactor;
mod param;
mod solver;
mod util;

use std::collections::HashSet;

use crate::def::*;
use crate::interactor::*;
use crate::param::*;
use crate::solver::*;
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

fn get_query_count(input: &Input) -> usize {
    let query_limit = input.n.pow(2) * 2;
    let n = input.n as f64;
    let m = input.m as f64;
    let eps = input.eps;
    let dense = input.minos.iter().map(|mino| mino.len()).sum::<usize>() as f64 / (n * n);

    let pred = query_count_linear_regression(n, m, eps, dense);
    let pred = pred * query_limit as f64;

    pred.round().round() as usize
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let time_limit = 2.8;
    let query_limit = input.n.pow(2) * 2;
    let k = input.n * 2; // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut checked_s = HashSet::new();
    let mut v_history = vec![];

    let base_query_count = get_query_count(input).clamp(20, query_limit);
    eprintln!("base_query_count = {}", base_query_count);

    let steps: Vec<f64> = vec![0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        .into_iter()
        .filter(|x| x * (base_query_count as f64) < query_limit as f64)
        .collect();
    let step_sum: f64 = steps.iter().map(|x| x.sqrt()).sum();
    let step_ratio: Vec<f64> = steps.iter().map(|x| x.sqrt() / step_sum).collect();

    for i in 1..steps.len() {
        let query_count = (((steps[i] - steps[i - 1]) * base_query_count as f64).round() as usize)
            .clamp(0, query_limit - interactor.query_count - 5);
        let mut _queries = investigate(k, query_count, &v_history, &mut fixed, interactor, input);
        queries.extend(_queries);

        // ミノの配置を最適化
        let optimize_time_limit = if i < steps.len() - 1 {
            time_limit * step_ratio[i]
        } else {
            time_limit - time::elapsed_seconds()
        };
        let mut optimizer = MinoOptimizer::new(&queries, &input);
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

            vis_v(&v, answer);
            eprintln!("mino_loss:   {:10.5}", mino_loss);
            eprintln!("error_count: {}", error_count(&v, answer));
            eprintln!("query_count: {} / {}", interactor.query_count, query_limit);
            eprintln!("total_cost:  {:.5}", interactor.total_cost);

            if interactor.output_answer(&s) {
                vis_queries(&queries, &input);
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
