//! The outbound half of the training chain: hands the caller's data set to
//! `data_manipulate_api` and takes back the column-rescaled version.
//!
//! Training runs on the rescaled data, which is the whole point of the hop —
//! unscaled features force a learning rate small enough that convergence takes
//! millions of iterations. `ratios` comes back with the scaled data and is what
//! maps the resulting coefficients back to the caller's units.

use std::sync::OnceLock;

use axum::http::StatusCode;
use serde::Deserialize;

/// What `POST /manipulate-datas` answers with.
#[derive(Debug, Deserialize)]
pub struct ScaledData {
    /// One power of ten per feature column, with the exponent for `outputs`
    /// last, so the vector is `n + 1` long.
    pub ratios : Vec<usize>,
    pub inputs : Vec<Vec<f64>>,
    pub outputs : Vec<f64>,
}

/// Built once and reused: a fresh `Client` per request would throw away the
/// connection pool and open a new socket every time.
fn client() -> &'static reqwest::Client {
    static CLIENT : OnceLock<reqwest::Client> = OnceLock::new();

    CLIENT.get_or_init(reqwest::Client::new)
}

/// Defaults to loopback, where `data_manipulate_api` binds when it is not told
/// otherwise. Compose sets this to the service name instead.
fn endpoint() -> &'static str {
    static URL : OnceLock<String> = OnceLock::new();

    URL.get_or_init(|| {
        std::env::var("DATA_MANIPULATE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:3001/manipulate-datas".to_string())
    })
}

/// Sends the raw payload on untouched and returns the rescaled data set.
///
/// The payload is forwarded verbatim rather than being re-serialised from the
/// parsed struct, so the bytes the caller validated are the bytes that get
/// scaled.
///
/// A rejection from the scaler is passed back with its own status: the data set
/// it refuses is the one the caller sent, so its `400` is the caller's `400`.
/// Anything else — an unreachable service, a `5xx`, a body that will not
/// parse — becomes a `502`, because at that point the fault is this chain's,
/// not the caller's.
pub async fn rescale(dataset : String) -> Result<ScaledData, (StatusCode, String)> {
    let form = reqwest::multipart::Form::new().text("dataset", dataset);

    let response = client()
        .post(endpoint())
        .multipart(form)
        .send()
        .await
        .map_err(|error| (
            StatusCode::BAD_GATEWAY,
            format!("could not reach the scaling service at {}: {error}", endpoint())
        ))?;

    let status = response.status();
    let body = response.text().await.map_err(|error| (
        StatusCode::BAD_GATEWAY,
        format!("could not read the scaling service's reply: {error}")
    ))?;

    if status.is_client_error() {
        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_REQUEST),
            format!("the scaling service rejected the data set: {}", body.trim())
        ));
    }
    if !status.is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("the scaling service answered {status}: {}", body.trim())
        ));
    }

    serde_json::from_str(&body).map_err(|error| (
        StatusCode::BAD_GATEWAY,
        format!("could not parse the scaling service's reply: {error}")
    ))
}
