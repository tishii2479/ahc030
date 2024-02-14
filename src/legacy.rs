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

pub fn x_error(v: &Vec<Vec<f64>>, answer: &Option<Answer>) -> f64 {
    let Some(answer) = answer else { return 0. };
    let mut err = 0.;
    for i in 0..v.len() {
        for j in 0..v[i].len() {
            err += (v[i][j] - answer.v[i][j] as f64).powf(2.);
        }
    }
    err
}

pub fn vis_prob(x: &Vec<Vec<f64>>, answer: &Option<Answer>) {
    let has_answer = answer.is_some();
    for i in 0..x.len() {
        for j in 0..x[i].len() {
            let color_value = (x[i][j] * 256.).clamp(0., 255.) as usize;
            let color = format!("#FF{:02x}{:02x}", 255 - color_value, 255 - color_value);
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

fn action_move_two(
    &mut self,
    mino_is: &mut Vec<usize>,
    next_mino_poss: &mut Vec<(usize, usize)>,
) -> f64 {
    let mut score_diff = 0.;
    let r = 2; // :param、NOTE: 可変にできる
    let sample_size = 9; // :param
    let cand_size = 3; // :param

    while mino_is.len() < r {
        let mino_i = rnd::gen_range(0, self.input.m);
        if mino_is.contains(&mino_i) {
            continue;
        }
        mino_is.push(mino_i);
        score_diff += self.toggle_mino(mino_i, self.mino_pos[mino_i], false);
    }

    let mut evals: Vec<Vec<(f64, (usize, usize))>> = vec![vec![]; r];
    for (i, &mino_i) in mino_is.iter().enumerate() {
        for _ in 0..sample_size {
            let next_mino_pos = (
                rnd::gen_range(0, self.mino_range[mino_i].0),
                rnd::gen_range(0, self.mino_range[mino_i].1),
            );
            let eval = self.toggle_mino(mino_i, next_mino_pos, true);
            evals[i].push((eval, next_mino_pos));
            self.toggle_mino(mino_i, next_mino_pos, false);
        }
        evals[i].sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    let mut best_score_diff = 1e50;
    let mut best_ij = (0, 0);
    for i in 0..cand_size {
        for j in 0..cand_size {
            let mut a = score_diff;
            a += self.toggle_mino(mino_is[0], evals[0][i].1, true);
            a += self.toggle_mino(mino_is[1], evals[1][j].1, true);

            if a < best_score_diff {
                best_ij = (i, j);
                best_score_diff = a;
            }

            self.toggle_mino(mino_is[0], evals[0][i].1, false);
            self.toggle_mino(mino_is[1], evals[1][j].1, false);
        }
    }

    let (i, j) = best_ij;
    self.toggle_mino(mino_is[0], evals[0][i].1, true);
    self.toggle_mino(mino_is[1], evals[1][j].1, true);
    next_mino_poss.push(evals[0][i].1);
    next_mino_poss.push(evals[1][j].1);

    best_score_diff
}

pub fn investigate(
    k: usize,
    query_count: usize,
    v_history: &Vec<Vec<Vec<usize>>>,
    fixed: &mut Vec<Vec<bool>>,
    interactor: &mut Interactor,
    input: &Input,
) -> Vec<(Vec<(usize, usize)>, f64)> {
    let mut queries = Vec::with_capacity(query_count);

    let mut mean = vec![vec![0.; input.n]; input.n];
    for i in 0..input.n {
        for j in 0..input.n {
            for t in 0..v_history.len() {
                for (di, dj) in D {
                    let (ni, nj) = (i + di, j + dj);
                    if ni < input.n && nj < input.n && v_history[t][ni][nj] > 0 {
                        mean[i][j] += 1.;
                        break;
                    }
                }
            }
            mean[i][j] /= v_history.len() as f64;
        }
    }
    let mut var = vec![vec![0.; input.n]; input.n];
    let mut var_sum = 0.;
    for i in 0..input.n {
        for j in 0..input.n {
            for t in 0..v_history.len() {
                var[i][j] += (v_history[t][i][j] as f64 - mean[i][j]).powf(2.);
            }
            var[i][j] /= v_history.len() as f64;
            var[i][j] = var[i][j].max(0.1);
            var_sum += var[i][j];
        }
    }

    let alpha = 10000. / var_sum;

    let mut prob_v = Vec::with_capacity(11000);
    for i in 0..input.n {
        for j in 0..input.n {
            let cnt = (var[i][j] * alpha).round().max(1.) as usize;
            prob_v.extend(vec![(i, j); cnt]);
        }
    }

    for _ in 0..query_count {
        let mut s = vec![];
        while s.len() < k {
            let a = prob_v[rnd::gen_range(0, prob_v.len())];
            if !s.contains(&a) && !fixed[a.0][a.1] {
                s.push(a);
            }
        }
        let obs_x = interactor.output_query(&s) as f64;
        let obs_x = ((obs_x - k as f64 * input.eps) / (1. - 2. * input.eps)).max(0.); // NOTE: 本当にあってる？
        queries.push((s, obs_x));
    }

    queries
}
