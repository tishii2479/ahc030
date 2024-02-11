use crate::def::*;
use crate::interactor::*;

pub mod rnd {
    static mut S: usize = 88172645463325252;

    #[inline]
    #[allow(unused)]
    pub fn next() -> usize {
        unsafe {
            S = S ^ S << 7;
            S = S ^ S >> 9;
            S
        }
    }

    #[inline]
    #[allow(unused)]
    pub fn nextf() -> f64 {
        (next() & 4294967295) as f64 / 4294967296.
    }

    #[inline]
    #[allow(unused)]
    pub fn gen_range(low: usize, high: usize) -> usize {
        assert!(low < high);
        (next() % (high - low)) + low
    }

    #[allow(unused)]
    pub fn shuffle<I>(vec: &mut Vec<I>) {
        for i in 0..vec.len() {
            let j = gen_range(0, vec.len());
            vec.swap(i, j);
        }
    }
}

pub mod time {
    static mut START: f64 = -1.;

    #[allow(unused)]
    pub fn start_clock() {
        let _ = elapsed_seconds();
    }

    #[inline]
    #[allow(unused)]
    pub fn elapsed_seconds() -> f64 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        unsafe {
            if START < 0. {
                START = t;
            }
            t - START
        }
    }
}

// pub fn vis_queries(queries: &Vec<(Vec<(usize, usize)>, i64)>, input: &Input) {
//     let mut c = vec![vec![0; input.n]; input.n];
//     for (s, _) in queries.iter() {
//         for &(i, j) in s {
//             c[i][j] += 1;
//         }
//     }
//     for i in 0..input.n {
//         for j in 0..input.n {
//             eprint!("{:4}", c[i][j]);
//         }
//         eprintln!();
//     }
// }

pub fn vis_v(v: &Vec<Vec<usize>>, answer: &Option<Answer>) {
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
            eprint!("{:2}", v[i][j]);
            if has_answer {
                let ans_v = answer.as_ref().unwrap().v[i][j];
                eprint!("/{}", ans_v);
            }
            eprint!("\x1b[m ");
        }
        eprintln!();
    }
}

pub fn get_s(v: &Vec<Vec<usize>>) -> Vec<(usize, usize)> {
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

pub fn get_v(
    mino_pos: &Vec<(usize, usize)>,
    minos: &Vec<Vec<(usize, usize)>>,
    n: usize,
) -> Vec<Vec<usize>> {
    let mut v = vec![vec![0; n]; n];
    for (pos, mino) in std::iter::zip(mino_pos, minos) {
        for (i, j) in mino {
            v[pos.0 + i][pos.1 + j] += 1;
        }
    }
    v
}

pub fn get_mino_range(minos: &Vec<Vec<(usize, usize)>>) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(minos.len());
    for mino in minos {
        let i_max = mino.iter().map(|&x| x.0).max().unwrap();
        let j_max = mino.iter().map(|&x| x.1).max().unwrap();
        ranges.push((i_max, j_max));
    }
    ranges
}

pub fn error_count(v: &Vec<Vec<usize>>, answer: &Option<Answer>) -> i64 {
    let Some(answer) = answer else { return 0 };
    let mut error_count = 0;
    for i in 0..v.len() {
        for j in 0..v[i].len() {
            if answer.v[i][j] != v[i][j] {
                error_count += 1;
            }
        }
    }
    error_count
}

pub fn exit(interactor: &mut Interactor, input: &Input) {
    eprintln!(
        "params: n = {}, m = {}, eps = {:.2}",
        input.n, input.m, input.eps
    );
    eprintln!(
        "result: {{\"score\": {:.6}, \"duration\": {:.4}, \"query_count\": {}}}",
        interactor.total_cost,
        time::elapsed_seconds(),
        interactor.query_count,
    );
    std::process::exit(0);
}
