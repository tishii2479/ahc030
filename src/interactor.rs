use crate::def::*;
use std::io::{Stdin, Write};

use proconio::*;
pub struct Interactor {
    source: proconio::source::line::LineSource<std::io::BufReader<Stdin>>,
    pub total_cost: f64,
}

impl Interactor {
    pub fn new() -> Interactor {
        Interactor {
            source: proconio::source::line::LineSource::new(std::io::BufReader::new(
                std::io::stdin(),
            )),
            total_cost: 0.,
        }
    }

    pub fn read_input(&mut self) -> Input {
        input! { from &mut self.source, n: usize, m: usize, eps: f64};
        let mut minos = vec![];
        for _ in 0..m {
            input! { from &mut self.source, d: usize, mino: [(usize, usize); d] };
            minos.push(mino);
        }
        Input { n, m, eps, minos }
    }

    pub fn output_query(&mut self, s: &Vec<(usize, usize)>) -> i64 {
        print!("q {}", s.len());
        for (i, j) in s {
            print!(" {} {}", i, j);
        }
        println!();
        self.flush();
        self.total_cost += 1. / (s.len() as f64).sqrt();

        input! { from &mut self.source, x: i64 }
        x
    }

    pub fn output_answer(&mut self, s: &Vec<(usize, usize)>) -> bool {
        print!("a {}", s.len());
        for (i, j) in s {
            print!(" {} {}", i, j);
        }
        println!();
        self.flush();
        input! { from &mut self.source, t: usize }
        t == 1
    }

    fn flush(&self) {
        std::io::stdout().flush().unwrap();
    }
}
