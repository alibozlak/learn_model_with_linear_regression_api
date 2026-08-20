use axum::Json;
use axum::extract::{Multipart, Query};
use axum::http::StatusCode;
use crate::{data_manipulate_client, json_converter, train_params};
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

    // The hop that makes a sane learning rate possible: the descent runs on
    // single-digit columns instead of the caller's raw magnitudes.
    let scaled = data_manipulate_client::rescale(json_real_datas.clone()).await?;

    let n = scaled.inputs[0].len();

    if scaled.ratios.len() != n + 1 {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "the scaling service returned {} ratios for {n} feature(s), expected {}",
                scaled.ratios.len(), n + 1
            )
        ));
    }

    let initial_coefficients : Vec<f64> = vec![0.0; n + 1];
    let (non_scaled_inputs, non_scaled_outputs) : (Vec<Vec<f64>>, Vec<f64>)
        = json_converter::training_data_from_json(&json_real_datas).unwrap();

    let mut without_feature_scaling = WithoutFeatureScaling::new(
        non_scaled_inputs.clone(), non_scaled_outputs.clone(), initial_coefficients
    );
    let J_before_learning : f64 = without_feature_scaling.J();

    without_feature_scaling = WithoutFeatureScaling::new(
        scaled.inputs, scaled.outputs, without_feature_scaling.coefficients
    );

    let scaled_coefficients : Vec<f64>
        = without_feature_scaling.train_model(params.learning_rate, params.loop_count);
    let real_last_coefficients : Vec<f64> = convert_to_real_coefficients(scaled.ratios, scaled_coefficients);

    without_feature_scaling = WithoutFeatureScaling::new(
        non_scaled_inputs, non_scaled_outputs, real_last_coefficients.clone()
    );
    let J_after_learning : f64 = without_feature_scaling.J();

    // Everything is reported in the scaled space the descent ran in. `ratios`
    // travels with it so the caller can lift the coefficients back into the
    // units their data set was in before the scaler saw it.
    Ok(Json(TrainingResult {
        J_before_learning,
        J_after_learning,
        last_coefficients : real_last_coefficients
    }))
}

pub fn convert_to_real_coefficients(ratios : Vec<usize>, scaled_coefficients : Vec<f64>) -> Vec<f64> {
    let n = ratios.len() - 1;
    let mut real_last_coefficients : Vec<f64> = vec![0.0; n];
    for i in 0..(n+1) {
        real_last_coefficients[i]
            = scaled_coefficients[i] * 10.0_f64.powi(ratios[n+1] as i32 - ratios[i] as i32);
    }
    real_last_coefficients
}
