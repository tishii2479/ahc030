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
