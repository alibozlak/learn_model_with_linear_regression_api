use axum::extract::Multipart;
use axum::http::StatusCode;
use crate::json_converter;
use crate::learning_without_feature_scaling::WithoutFeatureScaling;

pub async fn train(mut multipart: Multipart) -> Result<String, (StatusCode, String)> {
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

    let (inputs, outputs) = json_converter::training_data_from_json(&json_real_datas)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let n : usize = inputs[0].len();

    let mut without_feature_scaling = WithoutFeatureScaling::new(
        inputs, outputs, vec![0.0; n + 1]
    );

    // FixMe : I have will gotten user learning rate and loop count
    let (last_coefficients, J_before_learning, J_after_learning) : (Vec<f64>, f64, f64)
        = without_feature_scaling.train_model(0.000003, 1_000_000);

    json_converter::coefficients_to_json(&last_coefficients, J_before_learning, J_after_learning)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))
}