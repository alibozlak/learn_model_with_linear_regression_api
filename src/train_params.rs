use axum::http::StatusCode;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TrainParams {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    #[serde(default = "default_loop_count")]
    pub loop_count: usize
}

/// Sized for the rescaled data the descent now runs on, not the caller's raw
/// magnitudes. Single-digit columns tolerate a step some four orders of
/// magnitude larger than unscaled ones did, and reach a lower cost in a
/// fiftieth of the iterations.
///
/// The ceiling is data-dependent — a column the scaler leaves alone, because it
/// is already below 10, still drives the curvature — so these sit well under
/// the point where the sample data set diverges rather than at it.
fn default_learning_rate() -> f64 { 0.01 }
fn default_loop_count() -> usize { 20_000 }

const MAX_LOOP_COUNT: usize = 1_000_000;

impl TrainParams {
    pub fn validate(&self) -> Result<(), (StatusCode, String)> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err((StatusCode::BAD_REQUEST, format!(
                "learning_rate must be a finite number greater than 0, got {}",
                self.learning_rate
            )));
        }
        if self.loop_count == 0 {
            return Err((StatusCode::BAD_REQUEST,
                        "loop_count must be at least 1".to_string()));
        }
        if self.loop_count > MAX_LOOP_COUNT {
            return Err((StatusCode::BAD_REQUEST, format!(
                "loop_count must not exceed {MAX_LOOP_COUNT}, got {}", self.loop_count
            )));
        }
        Ok(())
    }
}