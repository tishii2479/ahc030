fn optimize_v(queries: &Vec<(Vec<(usize, usize)>, f64)>, input: &Input) -> Vec<Vec<f64>> {
    let mut v = vec![vec![0.; input.n]; input.n];

    // 初期解を作る
    let poly_count = input.minos.iter().map(|mino| mino.len()).sum::<usize>();
    for _ in 0..poly_count {
        v[rnd::gen_range(0, input.n)][rnd::gen_range(0, input.n)] += 1.;
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

    const ITERATION: usize = 100000;
    for _t in 0..ITERATION {
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

fn optimize_mino_pos2(x: &Vec<Vec<f64>>, input: &Input) -> Vec<(usize, usize)> {
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
        let v = get_v(mino_pos, minos, x.len());

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

    // const START_TEMP: f64 = 1e2;
    // const END_TEMP: f64 = 1e-2;
    const ITERATION: usize = 100000;
    for _t in 0..ITERATION {
        let cur_score = calc_score(&mino_pos, &x, &input.minos);
        let mut mino_is = vec![];
        let mut prev_mino_poss = vec![];
        // 近傍の工夫
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
        let new_score = calc_score(&mino_pos, &x, &input.minos);
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
        calc_score(&mino_pos, x, &input.minos)
    );

    mino_pos
}
