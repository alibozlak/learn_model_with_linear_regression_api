use axum::Json;
use axum::extract::{Multipart, Query};
use axum::http::StatusCode;
use crate::{json_converter, train_params};
use crate::json_converter::TrainingResult;
use crate::learning_without_feature_scaling::WithoutFeatureScaling;

pub async fn train(
    Query(params) : Query<train_params::TrainParams>,
    mut multipart: Multipart
) -> Result<Json<TrainingResult>, (StatusCode, String)> {

    params.validate().map_err(|e| (e.0, e.1))?;

    let mut payload: Option<String> = None;

    while let Some(field) = multipart.next_field().await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        if field.name() == Some("dataset") {
            payload = Some(field.text().await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?);
        }
    }

    let json_real_datas = payload.ok_or((
        StatusCode::BAD_REQUEST,
        "Form hasn't a 'dataset' named field !!".to_string(),
    ))?;

    let (inputs, outputs, initial_coefficients) 
        = json_converter::training_data_from_json(&json_real_datas)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let n : usize = inputs[0].len();

    let mut initial_coefficients : Vec<f64> = initial_coefficients;
    if initial_coefficients.len() == 2 { initial_coefficients = vec![0.0; n+1] }

    let mut without_feature_scaling = WithoutFeatureScaling::new(
        inputs, outputs, initial_coefficients
    );

    let (last_coefficients, J_before_learning, J_after_learning) : (Vec<f64>, f64, f64)
        = without_feature_scaling.train_model(params.learning_rate, params.loop_count);

    // `Json` labels the reply `application/json` and serialises straight into the
    // response body. Returning a `String` built by `coefficients_to_json` sent
    // the same bytes, but under `text/plain`, and paid for an extra copy.
    Ok(Json(TrainingResult {
        last_coefficients,
        J_before_learning,
        J_after_learning
    }))
}