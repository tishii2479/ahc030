use crate::def::*;
use crate::interactor::*;

use itertools::iproduct;
use std::collections::HashSet;

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

pub fn exit(interactor: &mut Interactor) {
    eprintln!(
        "result: {{\"score\": {:.6}, \"duration\": {:.4}, \"query_count\": {}}}",
        interactor.total_cost,
        time::elapsed_seconds(),
        interactor.query_count,
    );
    std::process::exit(0);
}

pub fn add_delta(
    from_pos: (usize, usize),
    mino_range: (usize, usize),
    delta: (i64, i64),
) -> (usize, usize) {
    // TODO: 外れているならNoneを返す、現状は少し偏っている
    let ni = (from_pos.0 as i64 + delta.0).clamp(0, mino_range.0 as i64 - 1) as usize;
    let nj = (from_pos.1 as i64 + delta.1).clamp(0, mino_range.1 as i64 - 1) as usize;
    (ni, nj)
}

pub fn calc_error(y: f64, y_hat: f64, s_len: usize) -> f64 {
    fn adjusted_q_len(x: usize) -> f64 {
        if x == 1 {
            1e-1 // :param
        } else {
            x as f64 // :param
        }
    }
    (y - y_hat).powf(2.) / adjusted_q_len(s_len)
}

pub fn get_weighted_delta_using_neighbors(max_dist: i64) -> Vec<(i64, i64)> {
    let mut delta = vec![];
    let p = 2.; // :param
    for (di, dj) in iproduct!(-max_dist..=max_dist, -max_dist..=max_dist) {
        let dist = ((i64::abs(di) + i64::abs(dj)) as f64).max(1.);
        let cnt = ((max_dist as f64 * 2.).powf(p) / dist.powf(p))
            .round()
            .max(1.) as usize;
        delta.extend(vec![(di, dj); cnt]);
    }
    delta
}

pub fn get_weighted_delta_using_duplication(
    max_dist: i64,
    input: &Input,
) -> Vec<Vec<Vec<(i64, i64)>>> {
    let mut weighted_delta = vec![vec![vec![]; input.m]; input.m];
    for mino_i in 0..input.m {
        for mino_j in 0..input.m {
            if mino_i == mino_j {
                continue;
            }
            let set: HashSet<&(usize, usize)> = HashSet::from_iter(input.minos[mino_j].iter());
            for d in iproduct!(-max_dist..=max_dist, -max_dist..=max_dist) {
                let mut duplicate_count = 0;
                for &p in &input.minos[mino_i] {
                    let (i, j) = (p.0 as i64 + d.0, p.1 as i64 + d.1);
                    let v = (i as usize, j as usize);
                    if set.contains(&v) {
                        duplicate_count += 1;
                    }
                }
                weighted_delta[mino_i][mino_j].extend(vec![d; duplicate_count.max(1)]);
            }
        }
    }

    weighted_delta
}

pub fn get_mino_range(input: &Input) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(input.minos.len());
    for mino in input.minos.iter() {
        let i_max = mino.iter().map(|&x| x.0).max().unwrap();
        let j_max = mino.iter().map(|&x| x.1).max().unwrap();
        ranges.push((input.n - i_max, input.n - j_max));
    }
    ranges
}
