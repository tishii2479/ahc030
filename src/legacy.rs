fn solve(interactor: &mut Interactor, input: &Input) {
    const T: usize = 1;
    let mut answer_s = vec![];
    loop {
        for i in 0..input.n {
            let l = (0..input.n + 1)
                .filter(|&j| j == input.n || (j % 3 == 0 && j != input.n - 1))
                .collect::<Vec<usize>>();
            for k in 0..l.len() - 1 {
                let mut xs = vec![];
                for j in l[k]..l[k + 1] {
                    let mut a = 0;
                    for _ in 0..T {
                        let s = (l[k]..l[k + 1])
                            .filter(|tj| tj != &j)
                            .map(|tj| (i, tj))
                            .collect::<Vec<(usize, usize)>>();
                        let x = interactor.output_query(&s);
                        a += x;
                    }
                    xs.push(a as f64 / T as f64);
                }
                let x_row_sum = xs.iter().sum::<f64>() / (xs.len() as f64 - 1.);
                for j in l[k]..l[k + 1] {
                    let approx_x = x_row_sum - xs[j - l[k]];
                    if approx_x > 0.5 {
                        answer_s.push((i, j));
                    }
                    eprint!("{:8.5} ", approx_x);
                }
            }
            eprintln!();
        }
        eprintln!();
        vis_answer(&answer_s, input);
        if interactor.output_answer(&answer_s) {
            let score = (interactor.total_cost * 1e6).round() as i64;
            exit(score);
        }
    }
}

// // ランダムに調べる
// for _ in 0..input.n * input.n {
//     let mut s = vec![];
//     while s.len() < k * k {
//         let a = (rnd::gen_range(0, input.n), rnd::gen_range(0, input.n));
//         if !s.contains(&a) {
//             s.push(a);
//         }
//     }
//     let obs_x = interactor.output_query(&s) as f64;
//     let obs_x = ((obs_x - (k * k) as f64 * input.eps) / (1. - 2. * input.eps)).max(0.); // 補正
//     for &(ni, nj) in s.iter() {
//         x[ni][nj] += obs_x as f64 / (k * k) as f64;
//         c[ni][nj] += 1;
//     }
//     queries.push((s, obs_x));
// }
