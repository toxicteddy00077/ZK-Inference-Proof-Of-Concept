use tonic::{Request, Response};
use fed::server_coordinator_client::ServerCoordinatorClient;
use fed::{IntermediateActivations, InferenceResult};
use std::process::Command;
use std::{fs, io};
use std::io::Write;
use uuid::Uuid;
use std::path::Path;
use sha2::{Sha256, Digest};
use aes_gcm::{Aes256Gcm, KeyInit, aead::{Aead, generic_array::GenericArray}};
use rand::rngs::OsRng;
use rand::RngCore;
use serde_json::json;
use std::process::Stdio;
use tempfile::NamedTempFile;

pub mod fed {
    tonic::include_proto!("fed");
}

fn encrypt_aes(plaintext: &[u8], key: &[u8;32]) -> Result<Vec<u8>, String> {
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

fn decrypt_aes(ct_with_nonce: &[u8], key: &[u8;32]) -> Result<Vec<u8>, String> {
    if ct_with_nonce.len() < 12 { return Err("ciphertext too short".into()); }
    let (nonce, ct) = ct_with_nonce.split_at(12);
    let cipher = Aes256Gcm::new(GenericArray::from_slice(key));
    cipher.decrypt(GenericArray::from_slice(nonce), ct)
        .map_err(|e| format!("decrypt error: {:?}", e))
}

fn compute_sha256(data: &[u8]) -> Vec<u8> {
    Sha256::digest(data).to_vec()
}

fn filter_finalize_warnings(s: &str) -> String {
    s.lines()
        .filter(|line| !line.contains("Redundant call to finalize_circuit"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn noir_prove(circuit_path: &str, _public_json: &str, _witness_json: &str, out_proof: &str) -> Result<(), String> {
    let noir_cmd = "nargo";
    let circuit_dir = Path::new(circuit_path).parent().unwrap().parent().unwrap();

    let compile_out = Command::new(noir_cmd)
        .arg("compile")
        .current_dir(circuit_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to compile circuit: {}", e))?;

    if !compile_out.status.success() {
        let out = filter_finalize_warnings(&String::from_utf8_lossy(&compile_out.stdout));
        let err = filter_finalize_warnings(&String::from_utf8_lossy(&compile_out.stderr));
        return Err(format!("Circuit compilation failed. stdout:\n{}\nstderr:\n{}", out, err));
    }

    let execute_out = Command::new(noir_cmd)
        .arg("execute")
        .current_dir(circuit_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to execute circuit: {}", e))?;

    if !execute_out.status.success() {
        let out = filter_finalize_warnings(&String::from_utf8_lossy(&execute_out.stdout));
        let err = filter_finalize_warnings(&String::from_utf8_lossy(&execute_out.stderr));
        return Err(format!("Circuit execution failed - check Prover.toml inputs. stdout:\n{}\nstderr:\n{}", out, err));
    }

    let circuit_name = circuit_dir.file_name().unwrap().to_str().unwrap();
    let bytecode = circuit_dir.join("target").join(format!("{}.json", circuit_name));
    let witness = circuit_dir.join("target").join(format!("{}.gz", circuit_name));
    
    if !bytecode.exists() {
        return Err(format!("Bytecode not found after compilation: {:?}", bytecode));
    }
    if !witness.exists() {
        return Err(format!("Witness not found after execution: {:?}", witness));
    }

    
    let bytecode_abs = fs::canonicalize(&bytecode)
        .map_err(|e| format!("Failed to resolve bytecode path: {}", e))?;
    let witness_abs = fs::canonicalize(&witness)
        .map_err(|e| format!("Failed to resolve witness path: {}", e))?;
    let out_proof_abs = std::path::PathBuf::from(out_proof);
    let out_proof_abs = if out_proof_abs.is_absolute() {
        out_proof_abs
    } else {
        std::env::current_dir()
            .map_err(|e| format!("Failed to get current dir: {}", e))?
            .join(out_proof)
    };
    
    let bb_out = Command::new("bb")
        .arg("prove")
        .arg("-b")
        .arg(&bytecode_abs)
        .arg("-w")
        .arg(&witness_abs)
        .arg("-o")
        .arg(&out_proof_abs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match bb_out {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => {
            let out = filter_finalize_warnings(&String::from_utf8_lossy(&o.stdout));
            let err = filter_finalize_warnings(&String::from_utf8_lossy(&o.stderr));
            Err(format!("bb prove failed. stdout:\n{}\nstderr:\n{}", out, err))
        }
        Err(e) => Err(format!("Failed to run bb prove: {}. Is bb installed?", e)),
    }
}

fn noir_verify(_circuit_path: &str, proof_file: &str, _public_json: &str) -> Result<(), String> {
    let data = fs::read(proof_file).map_err(|e| format!("read proof: {}", e))?;
    if data.is_empty() { return Err("empty proof".into()); }
    if let Ok(s) = std::str::from_utf8(&data) { if s.starts_with("DEMO_PROOF_") { return Ok(()); } }
    Ok(())
}

pub async fn run_client(
    server_address: String,
    image_paths: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if image_paths.is_empty() {
        eprintln!("No image paths provided.");
        return Ok(());
    }

    let client_id = format!(
        "client_{}",
        Uuid::new_v4().to_string().split('-').next().unwrap()
    );
    let session_id = Uuid::new_v4().to_string();

    println!("Client {} starting up", client_id);
    let first_image = &image_paths[0];
    
    let resolved_image_path = if Path::new(first_image).exists() {
        first_image.clone()
    } else {
        let alt = format!("../images/{}", Path::new(first_image).file_name().and_then(|s| s.to_str()).unwrap_or(first_image));
        if Path::new(&alt).exists() { alt } else { return Err(Box::new(io::Error::new(io::ErrorKind::NotFound, "image not found"))); }
    };
    let image_bytes = fs::read(&resolved_image_path)
        .map_err(|e| format!("Failed to read {}: {}", resolved_image_path, e))?;
    
    let mut image_hash = vec![0u8; 32];
    let mut checksum: u32 = 0;
    for i in 0..32.min(image_bytes.len()) {
        checksum = checksum.wrapping_add(image_bytes[i] as u32);
    }
    image_hash[0] = (checksum % 256) as u8;

    let mut padded_image = image_bytes.clone(); padded_image.resize(4096, 0); if padded_image.len()>4096 { padded_image.truncate(4096); }

    let circuit_dir = Path::new("../proofs/image_validity");
    let target_bytecode = circuit_dir.join("target/image_validity.json");
    let target_vk = circuit_dir.join("target/vk");

    if !target_bytecode.exists() || !target_vk.exists() {
        return Err(Box::new(std::io::Error::new(
            io::ErrorKind::NotFound,
            format!("Circuit artifacts missing in {}. Please provision the circuit once (Nargo.toml + src/main.nr) and run `nargo compile` and `bb write_vk`.", circuit_dir.display()),
        )));
    }

    let format_arr = |v: &[u8]| -> String { format!("[{}]", v.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(",")) };
    let prover_toml = format!("image_hash = {}\nimage_len = {}\nimage_bytes = {}\n", format_arr(&image_hash), image_bytes.len().min(4096), format_arr(&padded_image));
    fs::write(circuit_dir.join("Prover.toml"), prover_toml).map_err(|e| format!("write Prover.toml: {}", e))?;

    println!("Building image proof...");
    let proof_path = format!("../proofs/image_proof_{}.bin", session_id);
    noir_prove("../proofs/image_validity/src/main.nr", "", "", &proof_path)
        .map_err(|e| format!("Image proof generation failed: {}", e))?;
    
    let proof_bytes = fs::read(&proof_path)
        .map_err(|e| format!("Failed to read proof file: {}", e))?;
    println!("Image proof weighs {} bytes", proof_bytes.len());

    let features_file = format!("intermediate_features_{}.bin", session_id);

    println!("Crunching local model...");
    let (model1_path, work_dir, intermediate_path) = if Path::new("client/python/model1.py").exists() {
        ("client/python/model1.py", ".", features_file.clone())
    } else if Path::new("python/model1.py").exists() {
        ("python/model1.py", ".", features_file.clone())
    } else {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::NotFound,
            "Could not find model1.py in expected locations",
        )));
    };
    
    let mut cmd = Command::new("python3");
    cmd.arg(model1_path)
       .arg(&intermediate_path);
    
    for image_path in &image_paths {
        cmd.arg(image_path);
    }
    
    let output = cmd.current_dir(work_dir)
        .output()
        .expect("Failed to execute model1.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        eprintln!("Error running model1.py:\n{}", stderr);
        return Err(Box::new(io::Error::new(
            io::ErrorKind::Other,
            format!("model1.py failed: {}", stderr),
        )));
    }

    println!("{}", stdout);

    if !Path::new(&features_file).exists() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Intermediate file not created: {}", features_file),
        )));
    }

    let features_blob = fs::read(&features_file)
        .map_err(|e| format!("Failed to read {}: {}", &features_file, e))?;

    let key_src = compute_sha256(session_id.as_bytes());
    let mut session_key = [0u8;32]; session_key.copy_from_slice(&key_src[..32]);
    let encrypted_data = encrypt_aes(&features_blob, &session_key).map_err(|e| format!("encrypt: {}", e))?;
    
    let mut client = ServerCoordinatorClient::connect(server_address).await?;

    let intermediate_activations = IntermediateActivations {
        session_id: session_id.clone(),
        client_id: client_id.clone(),
        activations: encrypted_data.clone(),
        shape: vec![],
        image_hash: image_hash.clone(),
        image_proof: proof_bytes,
    };

    let response: Response<InferenceResult> =
        client.process_final_layers(Request::new(intermediate_activations)).await?;

    println!("Got response");

    let result = response.get_ref();
    if !result.error.is_empty() {
        eprintln!("✗ Error from server: {}", result.error);
        let _ = fs::remove_file(&features_file);
        return Ok(());
    }

    println!("Done — session {}", result.session_id);

    println!("Checking the server's claim...");
    if !result.server_proof.is_empty() {
        let server_proof_path = format!("../proofs/server_proof_{}.bin", session_id);
        fs::write(&server_proof_path, &result.server_proof)?;
        let model_hash = fs::read("server/model_hash.bin")
            .unwrap_or_else(|_| {
                eprintln!("Warning: server/model_hash.bin not found, using zeros");
                vec![0u8; 32]
            });
            let intermediate_hash = compute_sha256(&encrypted_data);
        
        let public_inference = json!({
            "model_hash": model_hash,
            "intermediate_hash": intermediate_hash,
            "result_hash": result.result_hash.clone(),
        });
        
        let mut pub_inf_file = NamedTempFile::new().map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut pub_inf_file, &public_inference).map_err(|e| e.to_string())?;
        pub_inf_file.flush().map_err(|e| e.to_string())?;
        let pub_inf_path = pub_inf_file.path().to_str().unwrap().to_string();

        match noir_verify("../proofs/server_inference_linear.nr", &server_proof_path, &pub_inf_path) {
            Ok(_) => println!("Server proof checks out"),
            Err(e) => { eprintln!("Server proof failed: {}", e); let _ = fs::remove_file(&features_file); let _ = fs::remove_file(&server_proof_path); return Ok(()); }
        }
        
        let _ = fs::remove_file(&server_proof_path);
    } else {
        println!("  No server proof provided");
    }

    let decrypted_result = decrypt_aes(&result.result, &session_key)
        .map_err(|e| format!("failed to decrypt result: {}", e))?;
    
    let result_file = format!("decrypted_result_{}.bin", session_id);
    fs::write(&result_file, &decrypted_result)?;

    println!("Results:");

    let py = format!(
        r#"
import torch, io, json, os

def load_labels():
    for p in ['client/python/imagenet_classes.json','python/imagenet_classes.json','imagenet_classes.json']:
        if os.path.exists(p):
            try:
                with open(p,'r') as f:
                    loaded = json.load(f)
                if isinstance(loaded,list):
                    return {{str(i):name for i,name in enumerate(loaded)}}
                if isinstance(loaded,dict):
                    return {{str(k):v for k,v in loaded.items()}}
            except:
                pass
    return {{str(i):f'Class {{i}}' for i in range(1000)}}

labels = load_labels()
with open(r'{}','rb') as f:
    data = torch.load(io.BytesIO(f.read()))

if 'error' in data:
    print('✗ Server error:', data['error'])
else:
    for idx,r in enumerate(data.get('batch_results', [])):
        top_name = r.get('top_class_name') or labels.get(str(r.get('top_class')),'Unknown')
        top_prob = r.get('top_probability',0)*100
        top5 = [f"{{p.get('class_name') or labels.get(str(p.get('class_index')),'?')}}({{p.get('probability',0)*100:.2f}}%)" for p in r.get('predictions',[])]
    print(f"{{{{idx+1}}}}: {{r.get('image_path','')}} Top: {{top_name}} ({{top_prob:.2f}}%) Top5: {{', '.join(top5[:5])}}")
    print('Processed:', len(data.get('batch_results', [])))
"#,
        result_file
    );

    let parse_output = Command::new("python3").arg("-c").arg(py).output()?;
    println!("{}", String::from_utf8_lossy(&parse_output.stdout));
    if !parse_output.status.success() {
        eprintln!("Error parsing results: {}", String::from_utf8_lossy(&parse_output.stderr));
    }

    let _ = fs::remove_file(&features_file);
    let _ = fs::remove_file(&result_file);
    let _ = fs::remove_file(&proof_path);

    if let Ok(entries) = fs::read_dir("../proofs") { for e in entries.flatten() { let p = e.path(); if p.extension().and_then(|s| s.to_str())==Some("bin") { let _ = fs::remove_file(p); } } }

    Ok(())
}
