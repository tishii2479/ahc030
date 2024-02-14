mod def;
mod interactor;
mod solver;
mod util;

use std::collections::HashSet;

use crate::def::*;
use crate::interactor::*;
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
    let n = input.n as f64;
    let m = input.m as f64;
    let eps = input.eps;
    let dense = input.minos.iter().map(|mino| mino.len()).sum::<usize>() as f64 / (n * n);

    let pred = -0.27925 * n
        + 0.061257 * m
        + -0.21641 * eps
        + -0.73188 * dense
        + 0.012837 * n.powf(2.0)
        + -0.0061535 * m.powf(2.0)
        + 21.703 * eps.powf(2.0)
        + 1.4862 * dense.powf(2.0)
        + -8.6805e-06 * n.powf(2.0) * n.powf(2.0)
        + -8.7748e-08 * n.powf(2.0) / n.powf(2.0)
        + 0.0002174 * n.powf(2.0) * m.powf(2.0)
        + 0.0023964 * n.powf(2.0) / m.powf(2.0)
        + 0.0020257 * n.powf(2.0) * eps.powf(2.0)
        + 8.1896e-09 * n.powf(2.0) / eps.powf(2.0)
        + -0.00069424 * n.powf(2.0) * dense.powf(2.0)
        + 2.2957e-06 * n.powf(2.0) / dense.powf(2.0)
        + -0.00020927 * m.powf(2.0) * n.powf(2.0)
        + 0.88736 * m.powf(2.0) / n.powf(2.0)
        + 1.7599e-07 * m.powf(2.0) * m.powf(2.0)
        + 5.3135e-10 * m.powf(2.0) / m.powf(2.0)
        + 0.0024407 * m.powf(2.0) * eps.powf(2.0)
        + -3.2592e-08 * m.powf(2.0) / eps.powf(2.0)
        + 0.0017039 * m.powf(2.0) * dense.powf(2.0)
        + 1.3132e-05 * m.powf(2.0) / dense.powf(2.0)
        + 0.0020274 * eps.powf(2.0) * n.powf(2.0)
        + 995.15 * eps.powf(2.0) / n.powf(2.0)
        + 0.0024407 * eps.powf(2.0) * m.powf(2.0)
        + -57.625 * eps.powf(2.0) / m.powf(2.0)
        + -285.54 * eps.powf(2.0) * eps.powf(2.0)
        + 0.0 * eps.powf(2.0) / eps.powf(2.0)
        + -6.5471 * eps.powf(2.0) * dense.powf(2.0)
        + 0.162 * eps.powf(2.0) / dense.powf(2.0)
        + -0.00069415 * dense.powf(2.0) * n.powf(2.0)
        + -97.941 * dense.powf(2.0) / n.powf(2.0)
        + 0.0017039 * dense.powf(2.0) * m.powf(2.0)
        + 3.1771 * dense.powf(2.0) / m.powf(2.0)
        + -6.5471 * dense.powf(2.0) * eps.powf(2.0)
        + 4.0716e-07 * dense.powf(2.0) / eps.powf(2.0)
        + -0.89267 * dense.powf(2.0) * dense.powf(2.0)
        + 0.0 * dense.powf(2.0) / dense.powf(2.0)
        + 1.5934;
    let pred = pred * (input.n as f64).powf(2.) * 2.;

    (pred.round().round() as usize).clamp(10, input.n.pow(2) * 2)
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n.pow(2) * 2;
    let k = input.n * 2; // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut checked_s = HashSet::new();

    let base_query_count = get_query_count(input);
    let steps = vec![0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5];
    eprintln!("base_query_count = {}", base_query_count);

    for i in 1..steps.len() {
        let query_count = (((steps[i] - steps[i - 1]) * base_query_count as f64).round() as usize)
            .clamp(0, query_limit - interactor.query_count - 5);
        let mut _queries = investigate(k, query_count, &vec![], &mut fixed, interactor, input);
        queries.extend(_queries);

        // ミノの配置を最適化
        let mut optimizer = MinoOptimizer::new(&queries, &input);
        let mut cands =
            optimizer.optimize(time::elapsed_seconds() + 2.8 / steps.len() as f64, true);
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (mino_loss, mino_pos) in cands.iter().take(i) {
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

            if interactor.output_answer(&s) {
                exit(interactor);
            }

            checked_s.insert(s);
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

    loop {
        interactor.output_answer(&vec![(0, 0)]);
    }
}
