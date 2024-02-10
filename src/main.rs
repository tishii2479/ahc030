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

fn vis_prob(x: &Vec<Vec<f64>>, answer: &Option<Answer>) {
    let has_answer = answer.is_some();
    for i in 0..x.len() {
        for j in 0..x[i].len() {
            let color_value = (x[i][j] * 256.).clamp(0., 255.) as usize;
            let color = format!("#FF{:x}{:x}", 255 - color_value, 255 - color_value);
            println!("#c {} {} {}", i, j, color);
            eprint!(
                "\x1b[38;2;{};{};{}m",
                255,
                255 - color_value,
                255 - color_value
            );
            if has_answer {
                let v = answer.as_ref().unwrap().v[i][j];
                if v > 0 {
                    let v = (130 + v * 60).clamp(0, 255);
                    eprint!("\x1b[48;2;{};100;100m", v);
                }
            }
            eprint!("{:5.3}", x[i][j]);
            eprint!("\x1b[m ");
        }
        eprintln!();
    }
}

fn vis_v(v: &Vec<Vec<usize>>, answer: &Option<Answer>) {
    let has_answer = answer.is_some();
    for i in 0..v.len() {
        for j in 0..v[i].len() {
            if has_answer {
                if answer.as_ref().unwrap().v[i][j] > v[i][j] {
                    eprint!("\x1b[38;2;0;0;255m");
                } else if answer.as_ref().unwrap().v[i][j] < v[i][j] {
                    eprint!("\x1b[38;2;255;0;0m");
                }
            }
            if v[i][j] > 0 {
                eprint!("\x1b[48;2;0;0;255m");
            }
            eprint!("  {}", v[i][j]);
            if has_answer {
                eprint!("/{}", answer.as_ref().unwrap().v[i][j]);
            }
            eprint!("\x1b[m ");
        }
        eprintln!();
    }
}

fn get_s(v: &Vec<Vec<usize>>) -> Vec<(usize, usize)> {
    let mut s = Vec::with_capacity(v.len() * v.len());
    for i in 0..v.len() {
        for j in 0..v[i].len() {
            if v[i][j] > 0 {
                s.push((i, j));
            }
        }
    }
    s
}

fn get_v(
    mino_pos: &Vec<(usize, usize)>,
    x: &Vec<Vec<f64>>,
    minos: &Vec<Vec<(usize, usize)>>,
) -> Vec<Vec<usize>> {
    let mut v = vec![vec![0; x.len()]; x[0].len()];
    for (pos, mino) in std::iter::zip(mino_pos, minos) {
        for (i, j) in mino {
            v[pos.0 + i][pos.1 + j] += 1;
        }
    }
    v
}

fn get_mino_range(minos: &Vec<Vec<(usize, usize)>>) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(minos.len());
    for mino in minos {
        let i_max = mino.iter().map(|&x| x.0).max().unwrap();
        let j_max = mino.iter().map(|&x| x.1).max().unwrap();
        ranges.push((i_max, j_max));
    }
    ranges
}

fn investigate(
    interactor: &mut Interactor,
    input: &Input,
) -> (Vec<Vec<f64>>, Vec<(Vec<(usize, usize)>, f64)>) {
    let k = 3;
    let mut x = vec![vec![0.; input.n]; input.n];
    let mut c = vec![vec![0; input.n]; input.n];
    let mut queries = Vec::with_capacity((input.n - k).pow(2));

    // ランダムに調べる
    // for _ in 0..input.n {
    //     let mut s = vec![];
    //     while s.len() < k * k {
    //         let a = (rnd::gen_range(0, input.n), rnd::gen_range(0, input.n));
    //         if !s.contains(&a) {
    //             s.push(a);
    //         }
    //     }
    //     let obs_x = interactor.output_query(&s);
    //     let obs_x = ((obs_x as f64 - (k * k) as f64 * input.eps) / (1. - 2. * input.eps)).max(0.); // 補正
    //     for &(ni, nj) in s.iter() {
    //         x[ni][nj] += obs_x as f64 / (k * k) as f64;
    //         c[ni][nj] += 1;
    //     }
    //     queries.push((s, obs_x));
    // }

    for i in 0..=input.n - k {
        for j in 0..=input.n - k {
            let mut s = Vec::with_capacity(k * k);
            for ni in i..i + k {
                for nj in j..j + k {
                    s.push((ni, nj));
                }
            }
            let obs_x = interactor.output_query(&s);
            let obs_x =
                ((obs_x as f64 - (k * k) as f64 * input.eps) / (1. - 2. * input.eps)).max(0.); // 補正
            for &(ni, nj) in s.iter() {
                x[ni][nj] += obs_x as f64 / (k * k) as f64;
                c[ni][nj] += 1;
            }
            queries.push((s, obs_x));
        }
    }
    for i in 0..input.n {
        for j in 0..input.n {
            x[i][j] /= c[i][j] as f64;
        }
    }

    (x, queries)
}

fn optimize_v(
    x: &Vec<Vec<f64>>,
    queries: &Vec<(Vec<(usize, usize)>, f64)>,
    input: &Input,
) -> Vec<Vec<f64>> {
    let mut v = vec![vec![0.; input.n]; input.n];

    // 初期解を作る
    // NOTE: 局所解にはまる原因になるかも
    let mut cand = Vec::with_capacity(input.n * input.n);
    for i in 0..input.n {
        for j in 0..input.n {
            cand.push((x[i][j], (i, j)));
        }
    }
    cand.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // NOTE: 重複を考慮していない
    let poly_count = input.minos.iter().map(|mino| mino.len()).sum::<usize>();
    for i in 0..poly_count {
        v[cand[i].1 .0][cand[i].1 .1] += 1.;
    }

    fn calc_score(v: &Vec<Vec<f64>>, queries: &Vec<(Vec<(usize, usize)>, f64)>) -> f64 {
        let mut score = 0.;
        for (s, obs_x) in queries {
            let mut x = 0.;
            for &(i, j) in s {
                x += v[i][j];
            }
            score += (obs_x - x).powf(2.);
        }
        score
    }

    for _t in 0..1000000 {
        let from = (rnd::gen_range(0, input.n), rnd::gen_range(0, input.n));
        let r = rnd::nextf().min(v[from.0][from.1]);
        let cur_score = calc_score(&v, queries);
        let to = (rnd::gen_range(0, input.n), rnd::gen_range(0, input.n));
        v[from.0][from.1] -= r;
        v[to.0][to.1] += r;
        let new_score = calc_score(&v, queries);

        if new_score < cur_score {
            // adopt
            // eprintln!("{:3} {:10.5} -> {:10.5}", _t, cur_score, new_score);
        } else {
            v[from.0][from.1] += r;
            v[to.0][to.1] -= r;
        }
    }

    v
}

fn optimize_mino_pos(x: &Vec<Vec<f64>>, input: &Input) -> Vec<(usize, usize)> {
    let mino_range = get_mino_range(&input.minos);
    let mut mino_pos = Vec::with_capacity(input.m);
    for k in 0..input.m {
        mino_pos.push((
            rnd::gen_range(0, input.n - mino_range[k].0),
            rnd::gen_range(0, input.n - mino_range[k].1),
        ));
    }

    fn calc_score(
        mino_pos: &Vec<(usize, usize)>,
        x: &Vec<Vec<f64>>,
        minos: &Vec<Vec<(usize, usize)>>,
    ) -> f64 {
        let v = get_v(mino_pos, x, minos);

        let mut score = 0.;
        for i in 0..x.len() {
            for j in 0..x[i].len() {
                // if v[i][j] == 0 && x[i][j] > 0.2 {
                //     score -= x[i][j].sqrt();
                // } else if v[i][j] > 0 {
                //     score += x[i][j].sqrt();
                // }
                score += (x[i][j] - v[i][j] as f64).powf(2.);
            }
        }
        score
    }

    // const START_TEMP: f64 = 1e-5;
    // const END_TEMP: f64 = 1e-6;
    const ITERATION: usize = 100000;
    for _t in 0..ITERATION {
        let cur_score = calc_score(&mino_pos, &x, &input.minos);
        let mino_i = rnd::gen_range(0, input.m);
        let prev_mino_pos = mino_pos[mino_i];
        mino_pos[mino_i] = (
            rnd::gen_range(0, input.n - mino_range[mino_i].0),
            rnd::gen_range(0, input.n - mino_range[mino_i].1),
        );
        let new_score = calc_score(&mino_pos, &x, &input.minos);
        // let progress = _t as f64 / ITERATION as f64;
        // let temp = START_TEMP.powf(1. - progress) * END_TEMP.powf(progress);
        // let adopt = rnd::nextf() < (-(new_score - cur_score) / temp).exp();
        if new_score < cur_score {
            // adopt
            // eprintln!("{:3} {:10.5} -> {:10.5}", _t, cur_score, new_score);
        } else {
            mino_pos[mino_i] = prev_mino_pos;
        }
    }

    mino_pos
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    // 情報を集める
    let (x, queries) = investigate(interactor, input);
    vis_prob(&x, &answer);

    // 理想的な油田の使用状況を調べる
    let x = optimize_v(&x, &queries, input);
    vis_prob(&x, &answer);

    // ミノの位置を探索
    let mino_pos = optimize_mino_pos(&x, &input);

    let v = get_v(&mino_pos, &x, &input.minos);
    vis_v(&v, answer);
    let s = get_s(&v);
    interactor.output_answer(&s);
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
