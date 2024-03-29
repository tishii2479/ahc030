use crate::def::*;
use std::io::{Stdin, Write};

use crate::util::*;

use proconio::*;

pub struct Interactor {
    source: proconio::source::line::LineSource<std::io::BufReader<Stdin>>,
    pub total_cost: f64,
    pub query_count: usize,
    query_limit: usize,
}

impl Interactor {
    pub fn new() -> Interactor {
        Interactor {
            source: proconio::source::line::LineSource::new(std::io::BufReader::new(
                std::io::stdin(),
            )),
            total_cost: 0.,
            query_count: 0,
            query_limit: 0,
        }
    }

    pub fn read_input(&mut self) -> Input {
        input! { from &mut self.source, n: usize, m: usize, eps: f64};
        let mut minos = Vec::with_capacity(m);
        for _ in 0..m {
            input! {
                from &mut self.source,
                d: usize,
                mino: [(usize, usize); d]
            }
            minos.push(mino);
        }
        self.query_limit = 2 * n * n;
        Input { n, m, eps, minos }
    }

    pub fn read_answer(&mut self, input: &Input) -> Answer {
        input! {
            from &mut self.source,
            mino_pos: [(usize, usize); input.m],
            v: [[usize; input.n]; input.n]
        }
        Answer { mino_pos, v }
    }

    pub fn output_query(&mut self, s: &Vec<(usize, usize)>) -> i64 {
        self.check_query_limit();

        print!("q {}", s.len());
        for (i, j) in s {
            print!(" {} {}", i, j);
        }
        println!();
        self.flush();
        self.total_cost += 1. / (s.len() as f64).sqrt();
        self.query_count += 1;

        input! { from &mut self.source, x: i64 }
        x
    }

    pub fn output_answer(&mut self, s: &Vec<(usize, usize)>) -> bool {
        self.check_query_limit();

        print!("a {}", s.len());
        for (i, j) in s {
            print!(" {} {}", i, j);
        }
        println!();
        self.flush();
        self.query_count += 1;

        input! { from &mut self.source, t: usize }

        if t == 0 {
            self.total_cost += 1.;
        }

        t == 1
    }

    fn flush(&self) {
        std::io::stdout().flush().unwrap();
    }

    fn check_query_limit(&self) {
        if self.query_count >= self.query_limit {
            eprintln!("failed to determine...");
            eprintln!(
                "result: {{\"score\": {:.6}, \"duration\": {:.4}, \"query_count\": {}}}",
                1e3,
                time::elapsed_seconds(),
                self.query_count,
            );
            std::process::exit(0);
        }
    }
}
