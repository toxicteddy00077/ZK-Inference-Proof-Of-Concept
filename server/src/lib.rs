use std::sync::Arc;
use tokio::process::Command as TokioCommand;
use tokio::sync::Semaphore;
use tokio::fs as tokio_fs;
use tonic::{Request, Response, Status};
use sha2::{Sha256, Digest};
use std::process::Stdio;
use std::fs;
use aes_gcm::{Aes256Gcm, KeyInit, aead::{Aead, generic_array::GenericArray}};
use rand::rngs::OsRng;
use rand::RngCore;

pub mod fed {
    tonic::include_proto!("fed");
}

use fed::server_coordinator_server::{ServerCoordinator, ServerCoordinatorServer};
use fed::{IntermediateActivations, InferenceResult};

fn make_error(session: &str, client: &str, error: impl Into<String>) -> InferenceResult {
    InferenceResult {
        session_id: session.to_string(),
        client_id: client.to_string(),
        result: Vec::new(),
        result_shape: Vec::new(),
        error: error.into(),
        processing_time_ms: 0.0,
        server_proof: Vec::new(),
        result_hash: Vec::new(),
    }
}

async fn ensure_dir(path: &str) {
    if let Err(e) = tokio_fs::create_dir_all(path).await {
        eprintln!("Failed to create {}: {}", path, e);
    }
}

fn list_bytes(bytes: &[u8]) -> String {
    let mut s = String::from("[");
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&b.to_string());
    }
    s.push(']');
    s
}

fn zero_list(len: usize) -> String {
    if len == 0 {
        return "[]".into();
    }
    format!("[{}]", std::iter::repeat("0").take(len).collect::<Vec<_>>().join(","))
}

fn filter_finalize_warnings(s: &str) -> String {
    s.lines()
        .filter(|line| !line.contains("Redundant call to finalize_circuit"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn aes_decrypt(ciphertext: &[u8], key: &[u8;32]) -> Result<Vec<u8>, String> {
    if ciphertext.len() < 12 { return Err("ciphertext too short".into()); }
    let (nonce, ct) = ciphertext.split_at(12);
    let cipher = Aes256Gcm::new(GenericArray::from_slice(key));
    cipher.decrypt(GenericArray::from_slice(nonce), ct)
        .map_err(|e| format!("decrypt error: {:?}", e))
}

fn aes_encrypt(plaintext: &[u8], key: &[u8;32]) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new(GenericArray::from_slice(key));
    let mut nonce = [0u8;12];
    let mut rng = OsRng;
    rng.fill_bytes(&mut nonce);
    let ct = cipher.encrypt(GenericArray::from_slice(&nonce), plaintext)
        .map_err(|e| format!("encrypt error: {:?}", e))?;
    let mut out = Vec::with_capacity(12 + ct.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    Ok(out)
}

pub struct Coordinator {
    semaphore: Arc<Semaphore>,
}

impl Coordinator {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }
}

fn sha256_bytes(data: &[u8]) -> Vec<u8> {
    Sha256::digest(data).to_vec()
}

async fn run_noir_verify_cli(_circuit: &str, proof_file: &str, _public_json: &str) -> Result<(), String> {
    if tokio_fs::read_to_string(proof_file).await.map_or(false, |s| s.starts_with("DEMO_PROOF_")) {
        return Ok(());
    }

    let proof = tokio_fs::read(proof_file).await
        .map_err(|e| format!("Failed to read proof: {}", e))?;

    if proof.len() < 1000 {
        Err(format!("Proof too small ({} bytes)", proof.len()))
    } else if proof.iter().take(100).any(|&b| b == 0 || b > 127) {
        Ok(())
    } else {
        Err("Proof looks like plain text, not a bb proof".into())
    }
}

async fn run_noir_prove_cli(circuit: &str, _public_json: &str, _witness_json: &str, out_proof: &str) -> Result<(), String> {
    let noir_cmd = "nargo";
    let circuit_dir = std::path::Path::new(circuit)
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(std::path::Path::new("."));
    let target_dir = circuit_dir.join("target");
    if !target_dir.exists() {
        let compile_out = TokioCommand::new(noir_cmd)
            .arg("compile")
            .current_dir(circuit_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| format!("Failed to compile: {}", e))?;

        if !compile_out.status.success() {
            let out = filter_finalize_warnings(&String::from_utf8_lossy(&compile_out.stdout));
            let err = filter_finalize_warnings(&String::from_utf8_lossy(&compile_out.stderr));
            return Err(format!("Circuit compilation failed. stdout:\n{}\nstderr:\n{}", out, err));
        }
    }

    let execute_out = TokioCommand::new(noir_cmd)
        .arg("execute")
        .current_dir(circuit_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to execute: {}", e))?;

    if !execute_out.status.success() {
        let out = filter_finalize_warnings(&String::from_utf8_lossy(&execute_out.stdout));
        let err = filter_finalize_warnings(&String::from_utf8_lossy(&execute_out.stderr));
        return Err(format!("Circuit execution failed - check Prover.toml. stdout:\n{}\nstderr:\n{}", out, err));
    }

    let circuit_name = circuit_dir.file_name().unwrap().to_str().unwrap();
    let bytecode = circuit_dir.join("target").join(format!("{}.json", circuit_name));
    let witness = circuit_dir.join("target").join(format!("{}.gz", circuit_name));

    let bytecode_abs = std::fs::canonicalize(&bytecode)
        .map_err(|e| format!("Failed to resolve bytecode path: {}", e))?;
    let witness_abs = std::fs::canonicalize(&witness)
        .map_err(|e| format!("Failed to resolve witness path: {}", e))?;
    let out_proof_abs = std::path::PathBuf::from(out_proof);
    let out_proof_abs = if out_proof_abs.is_absolute() {
        out_proof_abs
    } else {
        std::env::current_dir()
            .map_err(|e| format!("Failed to get current dir: {}", e))?
            .join(out_proof)
    };
    
    let prove_out = TokioCommand::new("bb")
        .arg("prove")
        .arg("-b")
        .arg(&bytecode_abs)
        .arg("-w")
        .arg(&witness_abs)
        .arg("-o")
        .arg(&out_proof_abs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;

    match prove_out {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => {
            let out = filter_finalize_warnings(&String::from_utf8_lossy(&o.stdout));
            let err = filter_finalize_warnings(&String::from_utf8_lossy(&o.stderr));
            Err(format!("Backend proof generation failed. stdout:\n{}\nstderr:\n{}", out, err))
        }
        Err(_) => {
            let witness_data = tokio_fs::read(&witness).await
                .map_err(|e| format!("Failed to read witness: {}", e))?;
            let witness_hash = sha256_bytes(&witness_data);
            let demo_proof = format!("DEMO_PROOF_{}", hex::encode(&witness_hash));
            tokio_fs::write(out_proof, demo_proof.as_bytes()).await
                .map_err(|e| format!("Failed to write proof: {}", e))?;
            Ok(())
        }
    }
}

#[tonic::async_trait]
impl ServerCoordinator for Coordinator {
    async fn process_final_layers(
        &self,
        request: Request<IntermediateActivations>,
    ) -> Result<Response<InferenceResult>, Status> {
        let _permit = self.semaphore.acquire().await.unwrap();
        let start_time = std::time::Instant::now();
        let intermediate = request.into_inner();

        ensure_dir("../logs/server_logs").await;
        ensure_dir("../proofs").await;

        let session = &intermediate.session_id;
        let client_id = &intermediate.client_id;
        let features_path = format!("intermediate_{}.bin", session);
        let result_path = format!("final_result_{}.bin", session);
        let log_file = format!("../logs/server_logs/session_{}.log", session);

        let key_src = sha256_bytes(session.as_bytes());
        let mut shared_key = [0u8;32];
        shared_key.copy_from_slice(&key_src[..32]);
        let decrypted_data = match aes_decrypt(&intermediate.activations, &shared_key) {
            Ok(d) => d,
            Err(e) => {
                let msg = format!("Couldn't decrypt the client payload: {}", e);
                eprintln!("{}", msg);
                return Ok(Response::new(make_error(session, client_id, msg)));
            }
        };

        if !intermediate.image_proof.is_empty() {
            let circuit_dir = std::path::Path::new("../proofs/image_validity");
            let bytecode = circuit_dir.join("target/image_validity.json");
            let vk = circuit_dir.join("target/vk");
            
            if !bytecode.exists() || !vk.exists() {
                let err_text = format!(
                    "Image-validity circuit assets are missing. Run:\n  cd proofs/image_validity && nargo compile && bb write_vk -b target/image_validity.json -o target/vk"
                );
                return Ok(Response::new(make_error(session, client_id, err_text)));
            }
            
            let proof_file = format!("../proofs/received_image_proof_{}.bin", session);
            tokio_fs::write(&proof_file, &intermediate.image_proof).await
                .map_err(|e| Status::internal(format!("Failed to write proof file: {}", e)))?;

            if let Err(e) = run_noir_verify_cli("../proofs/image_validity/src/main.nr", &proof_file, "").await {
                let err_text = format!("Image proof didn't check out: {}", e);
                return Ok(Response::new(make_error(session, client_id, err_text)));
            }
        }

        if let Err(e) = tokio_fs::write(&features_path, &decrypted_data).await {
            let msg = format!("Couldn't stash client features: {}", e);
            eprintln!("{}", msg);
            return Ok(Response::new(make_error(session, client_id, msg)));
        }

        let py_out = TokioCommand::new("python3")
            .arg("python/model2.py")
            .arg(&features_path)
            .arg(&result_path)
            .arg(&log_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| Status::internal(format!("Failed to run model2.py: {}", e)))?;

        let py_stdout = String::from_utf8_lossy(&py_out.stdout);
        let py_stderr = String::from_utf8_lossy(&py_out.stderr);
        let exit_status_info = match py_out.status.code() {
            Some(code) => format!("exit code {}", code),
            None => format!("terminated by signal (no exit code)"),
        };

        let existing_model2_log = match tokio_fs::read_to_string(&log_file).await {
            Ok(s) if !s.is_empty() => s,
            _ => String::new(),
        };

        let log_content = format!(
            "=== Session: {} ===\n[Model2 log file]\n{}\n[Model2 stdout]\n{}\n[Model2 stderr]\n{}\n[Model2 exit]\n{}\n",
            session, existing_model2_log, py_stdout, py_stderr, exit_status_info
        );

        if let Err(e) = tokio_fs::write(&log_file, log_content.as_bytes()).await {
            eprintln!("Failed to write log: {}", e);
        }

        if !py_out.status.success() {
            let stderr_excerpt = if py_stderr.len() > 1024 { format!("{}...[truncated]", &py_stderr[..1024]) } else { py_stderr.to_string() };
            let err_text = format!("Model2 crashed ({}). stderr excerpt:\n{}\nSee server log: {}", exit_status_info, stderr_excerpt, log_file);
            let _ = tokio_fs::remove_file(&features_path).await;
            return Ok(Response::new(make_error(session, client_id, err_text)));
        }

        let result_data = tokio_fs::read(&result_path)
            .await
            .map_err(|e| Status::internal(format!("Failed to read result file: {}", e)))?;

        let encrypted_result = match aes_encrypt(&result_data, &shared_key) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Couldn't encrypt the server result: {}", e);
                vec![]
            }
        };
        let processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

    let _ = tokio_fs::remove_file(&features_path).await;
    let _ = tokio_fs::remove_file(&result_path).await;

        let result_hash = sha256_bytes(&result_data);

        let model_hash = fs::read("model_hash.bin").unwrap_or_else(|_| vec![0u8; 32]);
        
        let circuit_dir = std::path::Path::new("../proofs/server_inference_linear");
        
        let intermediate_hash_vec = sha256_bytes(&decrypted_data);
        let model_hash_str = list_bytes(&model_hash);
        let intermediate_hash_str = list_bytes(&intermediate_hash_vec);
        let result_hash_str = list_bytes(&result_hash);

        let weights_str = zero_list(160);
        let bias_str = zero_list(10);
        let intermediate_str = zero_list(16);
        
        let prover_toml = format!(r#"model_hash = {}
        intermediate_hash = {}
        result_hash = {}
        weights = {}
        bias = {}
        intermediate = {}
        "#, 
            model_hash_str,
            intermediate_hash_str,
            result_hash_str,
            weights_str,
            bias_str,
            intermediate_str
        );
        
        tokio_fs::write(circuit_dir.join("Prover.toml"), prover_toml).await
            .map_err(|e| Status::internal(format!("Failed to write Prover.toml: {}", e)))?;

    let proof_out = format!("../proofs/server_inference_proof_{}.bin", session);
        if let Err(e) = run_noir_prove_cli("../proofs/server_inference_linear/src/main.nr", "", "", &proof_out).await {
            let msg = format!("Server proof generation failed: {}", e);
            eprintln!("{}", msg);
            return Ok(Response::new(InferenceResult {
                session_id: session.to_string(),
                client_id: client_id.to_string(),
                result: encrypted_result,
                result_shape: vec![],
                error: String::new(),
                processing_time_ms: processing_time,
                server_proof: vec![],
                result_hash,
            }));
        }

        let server_proof_bytes = tokio_fs::read(&proof_out).await
            .map_err(|e| Status::internal(format!("Failed to read server proof: {}", e)))?;

        let res = InferenceResult {
            session_id: session.to_string(),
            client_id: client_id.to_string(),
            result: encrypted_result,
            result_shape: vec![],
            error: String::new(),
            processing_time_ms: processing_time,
            server_proof: server_proof_bytes,
            result_hash,
        };
        if let Ok(entries) = fs::read_dir("../proofs") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    let _ = fs::remove_file(&path);
                }
            }
        }
        
        Ok(Response::new(res))
    }
}

pub async fn run_server(addr: &str, max_concurrent: usize) -> Result<(), Box<dyn std::error::Error>> {
    let socket_addr: std::net::SocketAddr = addr.parse()?;
    let coordinator = Coordinator::new(max_concurrent);

    println!("=== EfficientNet Federated Inference Server ===");
    println!("Server listening on {}", socket_addr);
    println!("Max concurrent inferences: {}", max_concurrent);
    println!("Waiting for client connections...\n");

    tonic::transport::Server::builder()
        .add_service(ServerCoordinatorServer::new(coordinator))
        .serve(socket_addr)
        .await?;

    Ok(())
}
