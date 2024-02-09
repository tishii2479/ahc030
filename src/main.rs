mod def;
mod interactor;
mod util;

use crate::def::*;
use crate::interactor::*;
use crate::util::*;

fn exit(score: i64) {
    eprintln!(
        "result: {{\"score\": {}, \"duration\": {:.4}}}",
        score,
        time::elapsed_seconds(),
    );
    std::process::exit(0);
}

fn solve(interactor: &mut Interactor, input: &Input) {
    let mut v = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            // let s = (0..input.n)
            //     .filter(|tj| tj != &j)
            //     .map(|tj| (i, tj))
            //     .collect::<Vec<(usize, usize)>>();
            let x = interactor.output_query(&vec![(i, j)]);
            if x > 0 {
                v.push((i, j));
            }
        }
    }
    if interactor.output_answer(&v) {
        let score = (interactor.total_cost * 1e6).round() as i64;
        exit(score);
    }
}

fn main() {
    time::start_clock();
    let mut interactor = Interactor::new();
    let input = interactor.read_input();
    solve(&mut interactor, &input);
}
