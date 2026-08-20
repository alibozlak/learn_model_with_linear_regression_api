use axum::Json;
use axum::extract::{Multipart, Query};
use axum::http::StatusCode;
use crate::{data_manipulate_client, json_converter, train_params};
use crate::json_converter::TrainingResult;
use crate::learning_without_feature_scaling::WithoutFeatureScaling;

/// `POST /train`: trains a model on the `dataset` field of a multipart body and
/// answers with the coefficients it converged to, in the units the caller's own
/// data set was written in.
///
/// The hop out to `data_manipulate_api` is what makes a sane learning rate
/// possible — the descent runs on columns the scaler has divided down instead
/// of the caller's raw magnitudes — but none of it should be visible in the
/// reply any more, so the handler undoes the scaling on the way back out:
///
/// * the payload is parsed a second time here, unscaled, because both costs are
///   measured against the numbers the caller actually sent. The scaler only
///   hands back the rescaled copy, so a cost read off it would be comparable to
///   nothing but the other cost.
/// * the coefficients pass through [`convert_to_real_coefficients`] before they
///   are reported, which is the conversion the caller used to be handed
///   `ratios` to do.
///
/// That is why three models are built around a single descent: one on the
/// unscaled data at the origin for the cost before, one on the scaled data that
/// actually runs the descent, and one on the unscaled data holding the
/// converted coefficients for the cost after. Only the middle one trains; the
/// other two exist to be asked `J()` once.
#[allow(non_snake_case)]
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
    let J_before_learning : u128 = without_feature_scaling.J();

    without_feature_scaling = WithoutFeatureScaling::new(
        scaled.inputs, scaled.outputs, without_feature_scaling.coefficients
    );

    let scaled_coefficients : Vec<f64>
        = without_feature_scaling.train_model(params.learning_rate, params.loop_count);
    let real_last_coefficients : Vec<f64> = convert_to_real_coefficients(scaled.ratios, scaled_coefficients);

    without_feature_scaling = WithoutFeatureScaling::new(
        non_scaled_inputs, non_scaled_outputs, real_last_coefficients.clone()
    );
    let J_after_learning : u128 = without_feature_scaling.J();

    Ok(Json(TrainingResult {
        J_before_learning,
        J_after_learning,
        last_coefficients : real_last_coefficients
    }))
}

/// Lifts the coefficients the descent ended on out of the scaled space and back
/// into the units the caller's data set was written in.
///
/// The scaler divided every column by a power of ten, so the line the descent
/// fitted, `y / 10^r_y = sum of a'_j * ( x_j / 10^r_j ) + b'`, describes the
/// same line as
///
/// ```text
/// a_j = a'_j * 10^(r_y - r_j)      b = b' * 10^r_y
/// ```
///
/// `ratios` and `scaled_coefficients` are both `n + 1` long but they do not
/// line up: `ratios` ends with `r_y`, the exponent `outputs` was divided by,
/// while the coefficients end with the bias. The loop therefore covers the
/// weights only and the bias is converted after it — it belongs to no column of
/// its own, and reading its exponent out of the slot at index `n` would pair
/// `r_y` with itself and leave the bias multiplied by `10^0`.
///
/// The exponent is per column, which is what makes the conversion exact: the
/// scaler reads a column's power of ten off that column's first value and then
/// divides every row of it by the same amount, so one exponent really does
/// describe the whole column. What the first-row rule costs is conditioning
/// rather than accuracy — a column whose first value is small next to the rest
/// comes back barely scaled — and that is the descent's problem, not this
/// function's. See "Known limitations" in the README.
pub fn convert_to_real_coefficients(ratios : Vec<usize>, scaled_coefficients : Vec<f64>) -> Vec<f64> {
    let n = ratios.len() - 1;
    let mut real_last_coefficients : Vec<f64> = vec![0.0; n + 1];
    for j in 0..n {
        real_last_coefficients[j]
            = scaled_coefficients[j] * 10.0_f64.powi(ratios[n] as i32 - ratios[j] as i32);
    }
    real_last_coefficients[n] = scaled_coefficients[n] * 10.0_f64.powi(ratios[n] as i32);

    real_last_coefficients
}
