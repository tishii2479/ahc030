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

fn investigate(
    k: usize,
    query_count: usize,
    interactor: &mut Interactor,
    input: &Input,
) -> Vec<(Vec<(usize, usize)>, f64)> {
    let mut queries = Vec::with_capacity(query_count);

    for _ in 0..query_count {
        let mut s = vec![];
        while s.len() < k {
            let a = (rnd::gen_range(0, input.n), rnd::gen_range(0, input.n));
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

fn toggle_mino(
    query_cache: &mut Vec<f64>,
    queries: &Vec<(Vec<(usize, usize)>, f64)>,
    query_indices: &Vec<Vec<Vec<usize>>>,
    mino: &Vec<(usize, usize)>,
    mino_pos: (usize, usize),
    turn_on: bool,
) -> f64 {
    let mut score_diff = 0.;
    for &(_i, _j) in mino.iter() {
        let (i, j) = (mino_pos.0 + _i, mino_pos.1 + _j);
        for &q_i in query_indices[i][j].iter() {
            score_diff -= (query_cache[q_i] - queries[q_i].1).powf(2.);
            if turn_on {
                query_cache[q_i] += 1.;
            } else {
                query_cache[q_i] -= 1.;
            }
            score_diff += (query_cache[q_i] - queries[q_i].1).powf(2.);
        }
    }
    score_diff
}

fn optimize_mino_pos(
    queries: &Vec<(Vec<(usize, usize)>, f64)>,
    input: &Input,
) -> Vec<(usize, usize)> {
    let mino_range = get_mino_range(&input.minos);
    let mut mino_pos = Vec::with_capacity(input.m);
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

    // const START_TEMP: f64 = 1e2;
    // const END_TEMP: f64 = 1e-2;
    const ITERATION: usize = 100000; // NOTE: 伸ばすと結構上がる
    for _t in 0..ITERATION {
        let mut mino_is = vec![];
        let mut prev_mino_poss = vec![];

        let mut score_diff = 0.;
        let r = rnd::gen_range(2, 4.min(input.m + 1));

        // TODO: 近傍の工夫
        // 有効な場所を多くする
        for _ in 0..r {
            let mino_i = rnd::gen_range(0, input.m);
            let prev_mino_pos = mino_pos[mino_i];
            mino_pos[mino_i] = (
                rnd::gen_range(0, input.n - mino_range[mino_i].0),
                rnd::gen_range(0, input.n - mino_range[mino_i].1),
            );
            mino_is.push(mino_i);
            prev_mino_poss.push(prev_mino_pos);

            score_diff += toggle_mino(
                &mut query_cache,
                queries,
                &query_indices,
                &input.minos[mino_i],
                prev_mino_pos,
                false,
            );
            score_diff += toggle_mino(
                &mut query_cache,
                queries,
                &query_indices,
                &input.minos[mino_i],
                mino_pos[mino_i],
                true,
            );
        }
        // let progress = _t as f64 / ITERATION as f64;
        // let temp = START_TEMP.powf(1. - progress) * END_TEMP.powf(progress);
        // let adopt = rnd::nextf() < (-(new_score - cur_score) / temp).exp();
        let adopt = score_diff < 0.;
        if adopt {
            // eprintln!("{:3} {:10.5} -> {:10.5}", _t, score, score + score_diff);
            score += score_diff;
        } else {
            for i in (0..r).rev() {
                let mino_i = mino_is[i];
                score_diff += toggle_mino(
                    &mut query_cache,
                    queries,
                    &query_indices,
                    &input.minos[mino_i],
                    mino_pos[mino_i],
                    false,
                );
                mino_pos[mino_i] = prev_mino_poss[i];
                score_diff += toggle_mino(
                    &mut query_cache,
                    queries,
                    &query_indices,
                    &input.minos[mino_i],
                    prev_mino_poss[i],
                    true,
                );
            }
        }
    }
    eprintln!("optimize_mino_pos_error: {:10.5}", score);

    mino_pos
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let mut queries = vec![];
    let k = input.n;
    let query_count = input.n;
    loop {
        // 情報を集める
        let query_count = query_count.min(input.n * input.n * 2 - interactor.query_count);
        let _queries = investigate(k, query_count, interactor, input);
        queries.extend(_queries);

        // ミノの配置を最適化
        let mino_pos = optimize_mino_pos(&queries, &input);
        let v = get_v(&mino_pos, &input.minos, input.n);
        vis_v(&v, answer);

        eprintln!("error_count: {}", error_count(&v, answer));

        let s = get_s(&v);
        interactor.output_answer(&s);
    }
}

fn main() {
    time::start_clock();
    let mut interactor = Interactor::new();
    let input = interactor.read_input();
    eprintln!("m = {}, eps = {:.2}", input.m, input.eps);
    let answer = if cfg!(feature = "local") {
        Some(interactor.read_answer(&input))
    } else {
        None
    };
    solve(&mut interactor, &input, &answer);
}
