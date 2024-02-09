mod def;
mod interactor;
mod util;

use crate::interactor::*;

fn main() {
    let mut interactor = Interactor::new();
    let input = interactor.read_input();
}
