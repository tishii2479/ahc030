use crate::def::*;
use std::io::{Stdin, Write};

use proconio::*;
pub struct Interactor {
    source: proconio::source::line::LineSource<std::io::BufReader<Stdin>>,
}

impl Interactor {
    pub fn new() -> Interactor {
        Interactor {
            source: proconio::source::line::LineSource::new(std::io::BufReader::new(
                std::io::stdin(),
            )),
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

    fn flush(&self) {
        std::io::stdout().flush().unwrap();
    }
}
