use std::env;
use std::error::Error;

use federated_gpt2_client::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Example: cargo run -- http://127.0.0.1:50051 img1.jpg img2.jpg");
        return Ok(());
    }

    let server_address = args[1].clone();
    let image_paths: Vec<String> = args[2..].to_vec();

    run_client(server_address, image_paths).await
}
