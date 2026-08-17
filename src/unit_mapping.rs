//! Moves coefficients between the caller's units and the rescaled space the
//! descent runs in.
//!
//! `data_manipulate_api` divides every value in feature column `j` by `10^r_j`
//! and every output by `10^r_y`, so the fitted line in scaled space
//!
//! ```text
//! y / 10^r_y = sum_j a'_j * (x_j / 10^r_j) + b'
//! ```
//!
//! rearranges into the caller's units as
//!
//! ```text
//! y = sum_j ( a'_j * 10^(r_y - r_j) ) * x_j + b' * 10^r_y
//! ```
//!
//! which is the pair of conversions below. Both take `ratios` laid out the way
//! the scaler returns it: one exponent per feature, the one for `outputs` last.

/// Scaled-space coefficients to the caller's units — applied to what the
/// descent produces.
pub fn to_original_units(scaled : &[f64], ratios : &[usize]) -> Vec<f64> {
    let n = ratios.len() - 1;
    let output_exponent = ratios[n] as i32;

    let mut original : Vec<f64> = (0..n)
        .map(|j| scaled[j] * 10.0_f64.powi(output_exponent - ratios[j] as i32))
        .collect();
    original.push(scaled[n] * 10.0_f64.powi(output_exponent));

    original
}

/// The inverse, applied to `initial_coefficients` on the way in.
///
/// Without it a caller's starting point would be read as though it belonged to
/// the scaled data, putting the descent nowhere near where they asked it to
/// begin. Running it here is also what keeps a run resumable: the answer comes
/// back in the caller's units, so feeding it in again lands on exactly the
/// point the previous run stopped at.
pub fn to_scaled_units(original : &[f64], ratios : &[usize]) -> Vec<f64> {
    let n = ratios.len() - 1;
    let output_exponent = ratios[n] as i32;

    let mut scaled : Vec<f64> = (0..n)
        .map(|j| original[j] * 10.0_f64.powi(ratios[j] as i32 - output_exponent))
        .collect();
    scaled.push(original[n] * 10.0_f64.powi(-output_exponent));

    scaled
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two directions have to compose back to where they started, or a
    /// resumed run would drift away from the point it stopped at.
    #[test]
    fn round_trip_returns_the_original_coefficients() {
        let ratios = vec![2, 0, 1, 6];
        let original = vec![386.9, -210.6, 42.5, 764.9];

        let there_and_back = to_original_units(&to_scaled_units(&original, &ratios), &ratios);

        for (before, after) in original.iter().zip(there_and_back.iter()) {
            assert!((before - after).abs() < 1e-9, "{before} became {after}");
        }
    }

    /// A model fitted in scaled space has to predict the same value as its
    /// mapped-back twin does on the unscaled sample.
    #[test]
    fn mapped_coefficients_predict_the_same_value() {
        let ratios = vec![2, 0, 4];
        let sample = [110.0, 3.0];
        let scaled_sample = [110.0 / 100.0, 3.0];
        let scaled_coefficients = [1.5, 0.25, 0.75];

        let scaled_prediction : f64 = scaled_coefficients[0] * scaled_sample[0]
            + scaled_coefficients[1] * scaled_sample[1]
            + scaled_coefficients[2];

        let original = to_original_units(&scaled_coefficients, &ratios);
        let prediction : f64 = original[0] * sample[0] + original[1] * sample[1] + original[2];

        // The scaled model predicts a scaled output, so it has to be lifted by
        // the outputs' own exponent before the two can be compared.
        let lifted = scaled_prediction * 10.0_f64.powi(ratios[2] as i32);

        assert!((lifted - prediction).abs() < 1e-6, "{lifted} vs {prediction}");
    }
}
