use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::post;

mod learning_without_feature_scaling;
mod json_converter;
mod train_endpoint;

#[tokio::main]
async fn main() {

    let app = Router::new()
        .route("/train", post(train_endpoint::train))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app).await.unwrap();

    //1_000_000 iterations, learning_rate = 0.000003. Response :
    //{
    //  "last_coefficients":[382.42810776151464,-226.37674378439436,1100.1919909133364],
    //  "J_before_learning":1_567_635_000.0,
    //  "J_after_learning":1_465_777.0
    // }
}
