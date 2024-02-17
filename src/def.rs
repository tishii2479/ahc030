pub struct Input {
    pub n: usize,
    pub m: usize,
    pub eps: f64,
    pub minos: Vec<Vec<(usize, usize)>>,
}

pub struct Answer {
    pub mino_pos: Vec<(usize, usize)>,
    pub v: Vec<Vec<usize>>,
}
