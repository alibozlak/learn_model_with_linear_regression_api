use axum::Json;
use axum::extract::{Multipart, Query};
use axum::http::StatusCode;
use crate::{data_manipulate_client, json_converter, train_params, unit_mapping};
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

    // Parsed here as well as by the scaler, because `initial_coefficients` is
    // the caller's and does not come back from the hop, and because the checks
    // this runs decide what the caller is told when the payload is malformed.
    let (inputs, _outputs, initial_coefficients)
        = json_converter::training_data_from_json(&json_real_datas)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let n : usize = inputs[0].len();

    let mut initial_coefficients : Vec<f64> = initial_coefficients;
    if initial_coefficients.len() == 2 { initial_coefficients = vec![0.0; n+1] }

    // The hop that makes a sane learning rate possible: the descent runs on
    // single-digit columns instead of the caller's raw magnitudes.
    let scaled = data_manipulate_client::rescale(json_real_datas).await?;

    if scaled.ratios.len() != n + 1 {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "the scaling service returned {} ratios for {n} feature(s), expected {}",
                scaled.ratios.len(), n + 1
            )
        ));
    }

    // The starting point arrives in the caller's units and has to be carried
    // into scaled space, or the descent would begin somewhere they never asked
    // for.
    let scaled_initial = unit_mapping::to_scaled_units(&initial_coefficients, &scaled.ratios);

    let mut without_feature_scaling = WithoutFeatureScaling::new(
        scaled.inputs, scaled.outputs, scaled_initial
    );

    let (scaled_coefficients, J_before_learning, J_after_learning) : (Vec<f64>, f64, f64)
        = without_feature_scaling.train_model(params.learning_rate, params.loop_count);

    Ok(Json(TrainingResult {
        // Handed back in the caller's units, which is both what is useful to
        // them and what makes the answer valid as the next run's
        // `initial_coefficients`.
        last_coefficients : unit_mapping::to_original_units(&scaled_coefficients, &scaled.ratios),
        J_before_learning,
        J_after_learning,
        ratios : scaled.ratios,
        scaled_last_coefficients : scaled_coefficients
    }))
}
