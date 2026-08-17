use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderValue, Method, header};
use axum::routing::post;
use tower_http::cors::CorsLayer;

mod learning_without_feature_scaling;
mod json_converter;
mod data_manipulate_client;
mod train_endpoint;
mod train_params;
mod unit_mapping;

#[tokio::main]
async fn main() {

    let app = Router::new()
        .route("/train", post(train_endpoint::train))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        // Outermost, so a preflight is answered before the body limit or the
        // router ever see the request.
        .layer(cors_layer());

    // Loopback by default: this is the only service the browser talks to, and
    // reaching it from another machine is not part of the local setup. The
    // image overrides it, because a container bound to loopback is unreachable
    // even from its own host.
    let bind_addr = std::env::var("BIND_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_string());

    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    println!("Listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app).await.unwrap();
}

/// A browser will not hand a cross-origin response to JavaScript unless the
/// reply carries `Access-Control-Allow-Origin`, so the Angular client on
/// another port cannot read a reply without this layer — the request itself
/// still reaches the handler, which is why the failure shows up as a status of
/// 0 on the client rather than as an error in the server log.
///
/// `curl` never sends a preflight and ignores these headers, so an endpoint
/// that works from the terminal can still be unreachable from the browser.
fn cors_layer() -> CorsLayer {
    let origins : Vec<HeaderValue> = std::env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| "http://localhost:4200".to_string())
        .split(',')
        .filter_map(|origin| origin.trim().parse().ok())
        .collect();

    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::POST])
        .allow_headers([header::CONTENT_TYPE])
}
