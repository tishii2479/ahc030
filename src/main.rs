mod def;
mod interactor;
mod util;

use crate::def::*;
use crate::interactor::*;
use crate::util::*;

fn vis_answer(s: &Vec<(usize, usize)>, input: &Input) {
    for i in 0..input.n {
        for j in 0..input.n {
            println!("#c {} {} white", i, j);
        }
    }
    for (i, j) in s {
        println!("#c {} {} red", i, j);
    }
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
                eprint!("\x1b[48;2;220;220;220m");
            }
            eprint!("{:6.4}", x[i][j]);
            eprint!("\x1b[m ");
        }
        eprintln!();
    }
}

fn solve(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let k = 3;
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

    interactor.output_answer(&vec![(0, 0)]);
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
