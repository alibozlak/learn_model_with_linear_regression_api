use axum::http::StatusCode;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct TrainParams {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_loop_count")]
    pub loop_count: usize,
}

fn default_learning_rate() -> f64 { 0.000003 }
fn default_loop_count() -> usize { 1_000_000 }

const MAX_LOOP_COUNT: usize = 5_000_000;

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