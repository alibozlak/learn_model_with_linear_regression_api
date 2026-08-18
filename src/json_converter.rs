//! Conversion between JSON payloads and the vector shapes the model works with.
//!
//! A training set is expected in this shape, where every entry of `inputs` is
//! one sample and the entry of `outputs` at the same index is that sample's
//! expected value:
//!
//! ```json
//! {
//!   "inputs":  [[55.0, 1.0], [130.0, 4.0]],
//!   "outputs": [23000.0, 48500.0]
//! }
//! ```
//!
//! and the result of a training run is written back as:
//!
//! ```json
//! { "scaled_last_coefficients": [375.13, -195.00, 1807.28] }
//! ```
//!
//! Nothing in this crate's `main` calls into this module — it exists to be
//! consumed from outside, so its items are exempted from the dead-code warning
//! a binary target would otherwise raise for them.
#![allow(dead_code)]

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A training set in the JSON shape described at the module level.
///
/// Exposed for callers that want the parsed struct itself; the usual entry
/// point is [`training_data_from_json`], which hands back the two plain vectors
/// the model's constructor takes.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingData {
    /// One entry per sample, each holding that sample's `n` feature values.
    pub inputs : Vec<Vec<f64>>,

    /// The expected output of the sample at the same index in [`Self::inputs`].
    pub outputs : Vec<f64>
}

/// The outcome of a training run.
///
/// Everything here is in the rescaled space the descent runs in. Mapping back
/// to the units the data set was in before scaling is the caller's to do, from
/// [`Self::ratios`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingResult {
    /// The cost before and after the run, measured in the rescaled space the
    /// descent actually minimises over.
    pub J_before_learning : f64,
    pub J_after_learning : f64,

    /// The power of ten each column was divided by, one per feature with the
    /// exponent for `outputs` last, as `data_manipulate_api` reported it.
    pub ratios : Vec<usize>,

    /// The coefficients the descent ended on, the bias `b` in the last slot,
    /// so the vector is `n + 1` long.
    pub scaled_last_coefficients : Vec<f64>,
}

/// Everything that can go wrong while converting between JSON and the model's
/// vectors.
///
/// The variants other than [`Self::Json`] all describe a payload that parses as
/// JSON but does not describe a usable training set. Catching them here means
/// the model's own constructor is never reached with data that would make it
/// panic — which matters when the JSON arrives from outside, e.g. as a request
/// body.
#[derive(Debug)]
pub enum JsonConverterError {
    /// The text was not valid JSON, or did not match the expected shape.
    Json(serde_json::Error),

    /// `inputs` was empty: there is nothing to train on, and no feature count
    /// can be derived.
    EmptyDataSet,

    /// There are not as many outputs as there are input samples.
    SampleCountMismatch { inputs : usize, outputs : usize },

    /// One sample carries a different number of features than the first one.
    RaggedSample { index : usize, expected : usize, found : usize }
}

impl fmt::Display for JsonConverterError {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid JSON payload: {error}"),
            Self::EmptyDataSet => write!(f, "\"inputs\" is empty, there is nothing to train on"),
            Self::SampleCountMismatch { inputs, outputs } => write!(
                f,
                "sample count mismatch: {inputs} input sample(s) but {outputs} output(s)"
            ),
            Self::RaggedSample { index, expected, found } => write!(
                f,
                "sample {index} has {found} feature(s) while the first sample has {expected}"
            )
        }
    }
}

impl Error for JsonConverterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            _ => None
        }
    }
}

impl From<serde_json::Error> for JsonConverterError {
    fn from(error : serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// Parses a training set and hands back the `(inputs, outputs)` pair that
/// `WithoutFeatureScaling::new` takes.
///
/// The data is validated first, so a successful return guarantees a non-empty
/// set, one output per sample, and the same feature count on every sample.
///
/// ```text
/// let json = r#"{"inputs": [[55.0, 1.0]], "outputs": [23000.0]}"#;
/// let (inputs, outputs) = json_converter::training_data_from_json(json)?;
/// let mut model = WithoutFeatureScaling::new(inputs, outputs, vec![0.0; 2]);
/// ```
pub fn training_data_from_json(
    json : &str
) -> Result<(Vec<Vec<f64>>, Vec<f64>), JsonConverterError> {
    let data : TrainingData = serde_json::from_str(json)?;
    validate(&data)?;

    Ok((data.inputs, data.outputs))
}

/// Serialises a training run's outcome into a single JSON line.
///
/// The endpoint no longer goes through this: it hands axum a
/// [`TrainingResult`] and lets `Json` serialise straight into the response
/// body. This stays for callers outside the HTTP path, which is why it takes
/// the whole struct rather than repeating its fields as arguments.
pub fn result_to_json(result : &TrainingResult) -> Result<String, JsonConverterError> {
    Ok(serde_json::to_string(result)?)
}

/// Same as [`result_to_json`], but indented for output a human reads or for a
/// file kept under version control.
pub fn result_to_json_pretty(result : &TrainingResult) -> Result<String, JsonConverterError> {
    Ok(serde_json::to_string_pretty(result)?)
}

/// Rejects payloads that parse as JSON but cannot describe a training set.
///
/// The feature count is taken from the first sample and every other sample is
/// checked against it, which is the check the model's own `validate_data_set`
/// does not perform.
fn validate(data : &TrainingData) -> Result<(), JsonConverterError> {
    let Some(first_sample) = data.inputs.first() else {
        return Err(JsonConverterError::EmptyDataSet);
    };

    if data.inputs.len() != data.outputs.len() {
        return Err(JsonConverterError::SampleCountMismatch {
            inputs : data.inputs.len(),
            outputs : data.outputs.len()
        });
    }

    let n : usize = first_sample.len();
    for (index, sample) in data.inputs.iter().enumerate() {
        if sample.len() != n {
            return Err(JsonConverterError::RaggedSample {
                index,
                expected : n,
                found : sample.len()
            });
        }
    }

    Ok(())
}