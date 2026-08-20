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
//! and the result of a training run is written back in this shape. The numbers
//! below are the ones the README's sample data set converges to under the
//! default parameters:
//!
//! ```json
//! {
//!   "J_before_learning": 1818750000,
//!   "J_after_learning":  883218,
//!   "last_coefficients": [
//!     245.86312190187883, 2566.50059820144, 7644.724306977841
//!   ]
//! }
//! ```
//!
//! `train_endpoint` reaches in here for [`training_data_from_json`], which is
//! how the handler gets hold of the unscaled vectors it measures both costs
//! against. The two serialisers are the part with no caller inside this binary
//! — they exist to be consumed from outside, and they are what the dead-code
//! exemption below is for.
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
/// Everything here is in the units the data set was written in, not the
/// rescaled space the descent actually ran in: the costs are measured against
/// the unscaled samples and the coefficients are converted back before they
/// reach this struct, so nothing about the scaling hop is left for the caller
/// to undo. The `ratios` that used to travel alongside them for exactly that
/// purpose are gone with it.
#[allow(non_snake_case)]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingResult {
    /// The cost at the origin and the cost the run ended on, both measured on
    /// the unscaled data set, so they are readable against the magnitudes in
    /// the payload rather than only against each other. An `after` that is not
    /// well below the `before` is the quickest sign that the learning rate was
    /// wrong for the data.
    ///
    /// Whole numbers: the mean squared error is summed as `f64` and truncated
    /// on the way out, which costs nothing at the magnitudes a real data set
    /// produces but reads as `0` for any fit whose cost falls below 1. A run
    /// that diverged also reports `0`, because the residuals are `NaN` by then
    /// — [`Self::last_coefficients`] comes back `null` and is the signal that
    /// can be trusted.
    pub J_before_learning : u128,
    pub J_after_learning : u128,

    /// The fit the descent converged to, as `[a_1, ..., a_n, b]` with the bias
    /// in the last slot, so the vector is `n + 1` long. Already lifted out of
    /// the scaled space by `train_endpoint::convert_to_real_coefficients`,
    /// which is what makes these coefficients applicable to the data set as it
    /// was sent.
    pub last_coefficients : Vec<f64>,
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