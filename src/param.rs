pub fn query_count_linear_regression(n: f64, m: f64, eps: f64, dense: f64) -> f64 {
    -0.27925 * n
        + 0.061257 * m
        + -0.21641 * eps
        + -0.73188 * dense
        + 0.012837 * n.powf(2.0)
        + -0.0061535 * m.powf(2.0)
        + 21.703 * eps.powf(2.0)
        + 1.4862 * dense.powf(2.0)
        + -8.6805e-06 * n.powf(2.0) * n.powf(2.0)
        + -8.7748e-08 * n.powf(2.0) / n.powf(2.0)
        + 0.0002174 * n.powf(2.0) * m.powf(2.0)
        + 0.0023964 * n.powf(2.0) / m.powf(2.0)
        + 0.0020257 * n.powf(2.0) * eps.powf(2.0)
        + 8.1896e-09 * n.powf(2.0) / eps.powf(2.0)
        + -0.00069424 * n.powf(2.0) * dense.powf(2.0)
        + 2.2957e-06 * n.powf(2.0) / dense.powf(2.0)
        + -0.00020927 * m.powf(2.0) * n.powf(2.0)
        + 0.88736 * m.powf(2.0) / n.powf(2.0)
        + 1.7599e-07 * m.powf(2.0) * m.powf(2.0)
        + 5.3135e-10 * m.powf(2.0) / m.powf(2.0)
        + 0.0024407 * m.powf(2.0) * eps.powf(2.0)
        + -3.2592e-08 * m.powf(2.0) / eps.powf(2.0)
        + 0.0017039 * m.powf(2.0) * dense.powf(2.0)
        + 1.3132e-05 * m.powf(2.0) / dense.powf(2.0)
        + 0.0020274 * eps.powf(2.0) * n.powf(2.0)
        + 995.15 * eps.powf(2.0) / n.powf(2.0)
        + 0.0024407 * eps.powf(2.0) * m.powf(2.0)
        + -57.625 * eps.powf(2.0) / m.powf(2.0)
        + -285.54 * eps.powf(2.0) * eps.powf(2.0)
        + 0.0 * eps.powf(2.0) / eps.powf(2.0)
        + -6.5471 * eps.powf(2.0) * dense.powf(2.0)
        + 0.162 * eps.powf(2.0) / dense.powf(2.0)
        + -0.00069415 * dense.powf(2.0) * n.powf(2.0)
        + -97.941 * dense.powf(2.0) / n.powf(2.0)
        + 0.0017039 * dense.powf(2.0) * m.powf(2.0)
        + 3.1771 * dense.powf(2.0) / m.powf(2.0)
        + -6.5471 * dense.powf(2.0) * eps.powf(2.0)
        + 4.0716e-07 * dense.powf(2.0) / eps.powf(2.0)
        + -0.89267 * dense.powf(2.0) * dense.powf(2.0)
        + 0.0 * dense.powf(2.0) / dense.powf(2.0)
        + 1.5934
}