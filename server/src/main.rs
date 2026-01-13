use std::error::Error;

// call library function in this crate
use federated_gpt2_server::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Address and concurrency can be adjusted or made CLI args if desired
    let addr = "0.0.0.0:50051";
    let max_concurrent = 32usize;

    run_server(addr, max_concurrent).await
}
