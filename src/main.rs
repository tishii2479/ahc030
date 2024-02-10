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
            if has_answer && answer.as_ref().unwrap().v[i][j] > 0 {
                eprint!("\x1b[48;2;210;100;100m");
            }
            eprint!("{:5.3}", x[i][j]);
            eprint!("\x1b[m ");
        }
        eprintln!();
    }
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    // 情報を集める
    let k = 4;
    let mut x = vec![vec![0.; input.n]; input.n];
    let mut c = vec![vec![0; input.n]; input.n];
    for i in 0..=input.n - k {
        for j in 0..=input.n - k {
            let mut s = vec![];
            for ni in i..i + k {
                for nj in j..j + k {
                    s.push((ni, nj));
                }
            }
            let nx = interactor.output_query(&s);
            for ni in i..i + k {
                for nj in j..j + k {
                    x[ni][nj] += nx as f64 / (k * k) as f64;
                    c[ni][nj] += 1;
                }
            }
        }
    }

    for i in 0..input.n {
        for j in 0..input.n {
            x[i][j] /= c[i][j] as f64;
        }
    }

    vis_prob(&x, &answer);

    // ミノの位置を探索
    let mino_range = calc_mino_range(&input.minos);
    let mut mino_pos = vec![];
    for k in 0..input.m {
        mino_pos.push((
            rnd::gen_range(0, input.n - mino_range[k].0),
            rnd::gen_range(0, input.n - mino_range[k].1),
        ));
    }

    fn calc_v(
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

    fn calc_score(
        mino_pos: &Vec<(usize, usize)>,
        x: &Vec<Vec<f64>>,
        minos: &Vec<Vec<(usize, usize)>>,
    ) -> f64 {
        let v = calc_v(mino_pos, x, minos);

        let mut score = 0.;
        for i in 0..x.len() {
            for j in 0..x[i].len() {
                // TODO: 0の時はペナルティを足す
                score += v[i][j] as f64 * x[i][j];
            }
        }
        score
    }

    for _ in 0..10000 {
        let cur_score = calc_score(&mino_pos, &x, &input.minos);
        let mino_i = rnd::gen_range(0, input.m);
        let prev_mino_pos = mino_pos[mino_i];
        // TODO: 近傍の工夫
        mino_pos[mino_i] = (
            rnd::gen_range(0, input.n - mino_range[mino_i].0),
            rnd::gen_range(0, input.n - mino_range[mino_i].1),
        );
        let new_score = calc_score(&mino_pos, &x, &input.minos);
        if new_score > cur_score {
            // adopt
        } else {
            mino_pos[mino_i] = prev_mino_pos;
        }
    }

    let v = calc_v(&mino_pos, &x, &input.minos);
    let mut s = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            if v[i][j] > 0 {
                s.push((i, j));
            }
        }
    }
    interactor.output_answer(&s);
}

fn calc_mino_range(minos: &Vec<Vec<(usize, usize)>>) -> Vec<(usize, usize)> {
    let mut ranges = vec![];
    for mino in minos {
        let i_max = mino.iter().map(|&x| x.0).max().unwrap();
        let j_max = mino.iter().map(|&x| x.1).max().unwrap();
        ranges.push((i_max, j_max));
    }
    ranges
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
