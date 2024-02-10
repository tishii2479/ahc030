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
    let mut queries = Vec::with_capacity((input.n - k).pow(2));

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

    fn calc_score(
        queries: &Vec<(Vec<(usize, usize)>, f64)>,
        mino_pos: &Vec<(usize, usize)>,
        minos: &Vec<Vec<(usize, usize)>>,
        n: usize,
    ) -> f64 {
        let v = get_v(mino_pos, minos, n);
        let mut score = 0.;
        for (s, obs_x) in queries {
            let mut x = 0.;
            for &(i, j) in s {
                x += v[i][j] as f64;
            }
            score += (obs_x - x).powf(2.);
        }
        score
    }

    // const START_TEMP: f64 = 1e2;
    // const END_TEMP: f64 = 1e-2;
    const ITERATION: usize = 100000;
    for _t in 0..ITERATION {
        let cur_score = calc_score(queries, &mino_pos, &input.minos, input.n);
        let mut mino_is = vec![];
        let mut prev_mino_poss = vec![];

        // TODO: 近傍の工夫
        // 有効な場所を多くする
        for _ in 0..2 {
            let mino_i = rnd::gen_range(0, input.m);
            let prev_mino_pos = mino_pos[mino_i];
            mino_pos[mino_i] = (
                rnd::gen_range(0, input.n - mino_range[mino_i].0),
                rnd::gen_range(0, input.n - mino_range[mino_i].1),
            );
            mino_is.push(mino_i);
            prev_mino_poss.push(prev_mino_pos);
        }
        let new_score = calc_score(queries, &mino_pos, &input.minos, input.n);
        // let progress = _t as f64 / ITERATION as f64;
        // let temp = START_TEMP.powf(1. - progress) * END_TEMP.powf(progress);
        // let adopt = rnd::nextf() < (-(new_score - cur_score) / temp).exp();
        let adopt = new_score < cur_score;
        if adopt {
            // adopt
            // eprintln!("{:3} {:10.5} -> {:10.5}", _t, cur_score, new_score);
        } else {
            for i in (0..2).rev() {
                mino_pos[mino_is[i]] = prev_mino_poss[i];
            }
        }
    }
    eprintln!(
        "optimize_mino_pos_error: {:10.5}",
        calc_score(queries, &mino_pos, &input.minos, input.n)
    );

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
