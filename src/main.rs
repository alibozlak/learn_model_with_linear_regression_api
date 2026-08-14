use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::post;

mod learning_without_feature_scaling;
mod json_converter;
mod train_endpoint;
mod train_params;

#[tokio::main]
async fn main() {

    let app = Router::new()
        .route("/train", post(train_endpoint::train))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app).await.unwrap();
}
