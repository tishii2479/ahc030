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

fn first_investigate(
    k: usize,
    r: usize,
    interactor: &mut Interactor,
    input: &Input,
) -> Vec<(Vec<(usize, usize)>, f64)> {
    let mut queries = Vec::with_capacity((input.n - r + 1).pow(2));

    let m = 2;
    for _ in 0..input.n {
        let mut s = vec![];
        let mut v = vec![];
        for _ in 0..m {
            let mut i;
            loop {
                i = rnd::gen_range(0, input.n);
                if !v.contains(&i) {
                    v.push(i);
                    break;
                }
            }
            for j in 0..input.n {
                s.push((i, j));
            }
        }
        let obs_x = interactor.output_query(&s) as f64;
        let obs_x = if k > 1 {
            ((obs_x - k as f64 * input.eps) / (1. - 2. * input.eps)).max(0.) // NOTE: 本当にあってる？
        } else {
            obs_x
        };
        queries.push((s, obs_x));
    }

    for _ in 0..input.n {
        let mut s = vec![];
        let mut v = vec![];
        for _ in 0..m {
            let mut j;
            loop {
                j = rnd::gen_range(0, input.n);
                if !v.contains(&j) {
                    v.push(j);
                    break;
                }
            }
            for i in 0..input.n {
                s.push((i, j));
            }
        }
        let obs_x = interactor.output_query(&s) as f64;
        let obs_x = if k > 1 {
            ((obs_x - k as f64 * input.eps) / (1. - 2. * input.eps)).max(0.) // NOTE: 本当にあってる？
        } else {
            obs_x
        };
        queries.push((s, obs_x));
    }

    queries
}

#[allow(unused)]
fn solve_answer_contains(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n.pow(2) * 2;
    let query_size = get_query_size(input); // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut v_history = vec![];

    let base_query_count = get_query_count(input).clamp(10, query_limit);
    eprintln!("base_query_count = {}", base_query_count);

    let mut _queries = investigate(
        query_size,
        (base_query_count / 4)
            .min(query_limit - interactor.query_count - 1)
            .max(1),
        &vec![],
        &mut fixed,
        interactor,
        input,
    );
    queries.extend(_queries);

    let mut optimizer = MinoOptimizer::new(None, &queries, &input);
    let mut cands = optimizer.optimize(time::elapsed_seconds() + 1.0, true);
    cands.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (_, mino_pos) = &cands[0];
    let v = get_v(&mino_pos, &input.minos, input.n);
    v_history.push(v);

    let mut _queries = investigate(
        query_size,
        (base_query_count / 2)
            .min(query_limit - interactor.query_count - 1)
            .max(1),
        &v_history,
        &mut fixed,
        interactor,
        input,
    );
    queries.extend(_queries);

    for _ in 0..1 {
        let mut sampled_queries = &queries;
        // let mut sampled_queries = vec![];
        // for _ in 0..queries.len() / 2 {
        //     sampled_queries.push(queries[rnd::gen_range(0, queries.len())].clone());
        // }

        let mut optimizer = MinoOptimizer::new(None, &sampled_queries, &input);
        let mut cands = optimizer.optimize(time::elapsed_seconds() + 2.0, true);
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

        eprintln!(
            "query_count: {}, cand sizes: {}",
            interactor.query_count,
            cands.len()
        );

        let mut checked_v = HashSet::new();
        for (mino_loss, mino_pos) in cands.iter() {
            let v = get_v(&mino_pos, &input.minos, input.n);
            let mut s = get_s(&v);
            s.sort();
            if checked_v.contains(&v) {
                continue;
            }
            if let Some(answer) = answer {
                let mut ans_s = get_s(&answer.v);
                ans_s.sort();
                if ans_s == s {
                    eprintln!("---------------- contained!!!!!! -----------------: ");
                    eprintln!(
                        "cand_i: {}, query_count: {}",
                        checked_v.len(),
                        interactor.query_count
                    );
                    eprintln!("answer_mino_loss: {:.5}", mino_loss);
                }

                if checked_v.len() == 0 {
                    eprintln!("mino_loss: {:.5}", mino_loss);
                    let mut cnt = 0;
                    for i in 0..input.m {
                        if mino_pos[i] != answer.mino_pos[i] {
                            cnt += 1;
                        }
                    }
                    eprintln!("miss_count: {}", cnt);
                }
            }

            if error_count(&v, answer) == 0 || checked_v.len() <= 5 {
                if interactor.output_answer(&s) {
                    exit(interactor);
                }
            }

            checked_v.insert(v);
        }

        if let Some(answer) = answer {
            let v = &answer.v;
            let mut score = 0.;
            for (s, x) in optimizer.queries.iter() {
                let mut v_sum = 0.;
                for &(i, j) in s.iter() {
                    v_sum += v[i][j] as f64;
                }
                score += calc_error(v_sum, *x, s.len());
            }
            eprintln!("ans_loss: {:.5}", score);
        }
        eprintln!("checked_v: {}", checked_v.len());
    }
}

#[allow(unused)]
fn solve_data_collection(interactor: &mut Interactor, input: &Input, answer: &Option<Answer>) {
    let query_limit = input.n.pow(2) * 2;
    let query_size = get_query_size(input); // :param

    let mut queries = vec![];
    let mut fixed = vec![vec![false; input.n]; input.n];
    let mut checked_s = HashSet::new();

    let mut query_count = 6;

    while interactor.query_count < query_limit {
        let next_query_count = ((query_count as f64 * 0.1).ceil() as usize).max(2);
        let mut _queries = investigate(
            query_size,
            next_query_count
                .min(query_limit - interactor.query_count - 1)
                .max(1),
            &vec![],
            &mut fixed,
            interactor,
            input,
        );
        query_count += next_query_count;
        queries.extend(_queries);

        // ミノの配置を最適化
        let mut optimizer = MinoOptimizer::new(None, &queries, &input);
        let mut cands = optimizer.optimize(time::elapsed_seconds() + 1.0, true);
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (mino_loss, mino_pos) in cands.iter().take(5) {
            let v = get_v(&mino_pos, &input.minos, input.n);
            let s = get_s(&v);
            if checked_s.contains(&s) {
                continue;
            }

            // vis_v(&v, answer);
            // eprint!("mino_loss:   {:10.5}", mino_loss);
            // eprintln!(", error_count: {}", error_count(&v, answer));
            // eprintln!("query_count: {} / {}", interactor.query_count, query_limit);
            // eprintln!("total_cost:  {:.5}", interactor.total_cost);

            if error_count(&v, answer) == 0 {
                if interactor.output_answer(&s) {
                    exit(interactor);
                }
            }

            checked_s.insert(s);
        }
    }
}

#[allow(unused)]
fn solve_greedy(interactor: &mut Interactor, input: &Input) {
    let mut s = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            let x = interactor.output_query(&vec![(i, j)]);
            if x > 0 {
                s.push((i, j));
            }
        }
    }
    assert!(interactor.output_answer(&s));
    exit(interactor);
}

fn investigate(
    k: usize,
    query_count: usize,
    v_history: &Vec<Vec<Vec<usize>>>,
    fixed: &mut Vec<Vec<bool>>,
    interactor: &mut Interactor,
    input: &Input,
) -> Vec<(Vec<(usize, usize)>, f64)> {
    const USE_HIGH_PROB: f64 = 0.5; // :param // NOTE: 徐々に大きくした方が良さそう
    let mut queries = Vec::with_capacity(query_count);

    let mut high_prob_v = vec![];
    if v_history.len() > 0 {
        for i in 0..input.n {
            for j in 0..input.n {
                for (di, dj) in D {
                    let (ni, nj) = (i + di, j + dj);
                    if ni < input.n && nj < input.n && v_history[v_history.len() - 1][ni][nj] > 0 {
                        high_prob_v.push((i, j));
                        break;
                    }
                }
            }
        }
    }

    for _ in 0..query_count {
        let mut s = vec![];
        while s.len() < k {
            let a = if high_prob_v.len() > 0 && rnd::nextf() < USE_HIGH_PROB {
                high_prob_v[rnd::gen_range(0, high_prob_v.len())]
            } else {
                (rnd::gen_range(0, input.n), rnd::gen_range(0, input.n))
            };
            if !s.contains(&a) && !fixed[a.0][a.1] {
                s.push(a);
            }
        }
        let obs_x = interactor.output_query(&s) as f64;
        let obs_x = if k > 1 {
            ((obs_x - k as f64 * input.eps) / (1. - 2. * input.eps)).max(0.) // NOTE: 本当にあってる？
        } else {
            obs_x
        };
        if s.len() == 1 {
            fixed[s[0].0][s[0].1] = true;
        }
        queries.push((s, obs_x));
    }

    queries
}
