use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct RequestOptions {
    method: String,
    path: String,
    body: Option<serde_json::Value>,
    headers: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiResponse {
    status: u16,
    data: Option<serde_json::Value>,
    headers: std::collections::HashMap<String, String>,
}

#[tauri::command]
fn request_python(options: RequestOptions) -> Result<ApiResponse, String> {
    let client = reqwest::blocking::Client::new();
    let url = format!("http://localhost:8000{}", options.path);
    
    let mut builder = match options.method.as_str() {
        "GET" => client.get(&url),
        "POST" => client.post(&url),
        "PUT" => client.put(&url),
        "DELETE" => client.delete(&url),
        _ => return Err("Invalid method".into()),
    };

    if let Some(h) = options.headers {
        for (k, v) in h {
            builder = builder.header(k, v);
        }
    }

    if let Some(body) = options.body {
        builder = builder.json(&body);
    }

    match builder.send() {
        Ok(res) => {
            let status = res.status().as_u16();
            let headers = res.headers()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                .collect();
            
            let data: Option<serde_json::Value> = res.json().ok();
            
            Ok(ApiResponse { status, data, headers })
        },
        Err(e) => Err(format!("Request failed: {}", e))
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // [OPTIMIZATION] Disable WebView2 GPU to free up RTX 4060 for backend
    std::env::set_var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "--disable-gpu --disable-d3d11");

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![request_python])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
