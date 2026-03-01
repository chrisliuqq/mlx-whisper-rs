# mlx-whisper-rs

MLX-accelerated Whisper speech recognition for Apple Silicon, written in pure Rust.

A faithful port of Apple's [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) Python library, using [`mlx-rs`](https://github.com/oxideai/mlx-rs) bindings to run on the Metal GPU via the MLX framework.

**Platform:** macOS on Apple Silicon (M1/M2/M3/M4) only.

---

## Features

- Full Whisper inference pipeline in Rust — no Python, no sidecar
- Loads any [mlx-community](https://huggingface.co/mlx-community) Whisper model directly from HuggingFace Hub
- Automatic language detection
- Sliding-window transcription for audio of any length
- Temperature fallback (retries with higher temperature on low-quality segments)
- `initial_prompt` support for domain-specific vocabulary
- Previous-text conditioning across segments
- In-memory audio input: parse WAV bytes or raw PCM without writing temp files
- MIT licensed

---

## Prerequisites

### 1. Install MLX via Homebrew

```bash
brew install mlx
```

This links against the system MLX library and avoids compiling MLX from source (which otherwise takes 10–20 minutes on first `cargo build`).

### 2. Install ffmpeg (required for `load_audio` only)

```bash
brew install ffmpeg
```

Only needed if you use `load_audio()` to load audio from a file path.
Not required if you use `audio_from_wav_bytes()` or `audio_from_pcm_s16le()`.

### 3. Copy tokenizer & filter assets

```bash
mkdir -p assets

# Copy from the mlx_whisper Python package
MLXW=$(python3 -c "import mlx_whisper, os; print(os.path.dirname(mlx_whisper.__file__))")
cp "$MLXW/assets/"*.tiktoken assets/
cp "$MLXW/assets/"*.npy assets/
```

The required files are:
- `assets/multilingual.tiktoken`
- `assets/gpt2.tiktoken`
- `assets/mel_filters_80.npy`
- `assets/mel_filters_128.npy`

---

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
mlx-whisper-rs = { git = "https://github.com/chrisliuqq/mlx-whisper-rs" }
# or as a local path dep:
# mlx-whisper-rs = { path = "../mlx-whisper-rs" }
```

### Transcribe a file

```rust
use std::path::PathBuf;
use mlx_whisper_rs::{
    audio::{load_audio, SAMPLE_RATE},
    load_models::load_model,
    transcribe::{transcribe, TranscribeOptions},
};

fn main() -> anyhow::Result<()> {
    let assets = PathBuf::from("assets");
    let mut model = load_model("mlx-community/whisper-large-v3-turbo")?;
    let audio = load_audio("audio.wav", SAMPLE_RATE)?;

    let result = transcribe(audio, &mut model, &assets, &TranscribeOptions {
        language: Some("zh".to_string()),
        ..Default::default()
    })?;

    println!("{}", result.text);
    Ok(())
}
```

### Transcribe WAV bytes (in-memory, no temp file)

```rust
use mlx_whisper_rs::audio::{audio_from_wav_bytes, SAMPLE_RATE};

let wav_bytes: Vec<u8> = std::fs::read("audio.wav")?;
let (audio, sample_rate) = audio_from_wav_bytes(&wav_bytes)?;
assert_eq!(sample_rate, SAMPLE_RATE as u32); // must be 16000 Hz
```

### Transcribe raw PCM bytes (e.g. from microphone ring buffer)

```rust
use mlx_whisper_rs::audio::audio_from_pcm_s16le;

// pcm_bytes: mono, 16kHz, s16le, no header
let audio = audio_from_pcm_s16le(&pcm_bytes);
```

### Run the CLI example

```bash
cargo run --example transcribe -- audio.wav --language zh --verbose
cargo run --example transcribe -- audio.wav --model mlx-community/whisper-large-v3-turbo
cargo run --example transcribe -- audio.wav  # auto-detect language
```

---

## API Reference

### `audio` module

```rust
// Load audio from file via ffmpeg → float32 mono 16kHz
pub fn load_audio(file: &str, sr: usize) -> Result<Array>

// Parse WAV bytes (RIFF/PCM) → (float32 mono Array, sample_rate)
// Supports: PCM s16le, PCM s32le, IEEE float32
// Multi-channel audio is averaged to mono
pub fn audio_from_wav_bytes(bytes: &[u8]) -> Result<(Array, u32)>

// Convert raw PCM s16le bytes (mono, no header) → float32 Array
pub fn audio_from_pcm_s16le(pcm_bytes: &[u8]) -> Array

// Compute log-Mel spectrogram (n_mels: 80 or 128)
pub fn log_mel_spectrogram(audio: Array, n_mels: usize, assets_dir: &Path) -> Result<Array>

// Trim or zero-pad to exact length along axis 0
pub fn pad_or_trim(array: Array, length: usize) -> Result<Array>

// Key constants
pub const SAMPLE_RATE: usize = 16000;
pub const N_FRAMES: usize = 3000;      // frames per 30-second chunk
pub const CHUNK_LENGTH: usize = 30;    // seconds per chunk
```

### `load_models` module

```rust
// Load a Whisper model from a local directory or HuggingFace repo ID
// Downloads automatically on first use, cached in ~/.cache/huggingface/
pub fn load_model(model_id: &str) -> Result<Whisper>
```

### `transcribe` module

```rust
pub struct TranscribeOptions {
    pub language: Option<String>,              // None = auto-detect
    pub task: String,                          // "transcribe" | "translate"
    pub verbose: bool,                         // print segments as decoded
    pub initial_prompt: Option<String>,        // vocabulary hint for Whisper
    pub condition_on_previous_text: bool,      // cross-segment context (default: true)
    pub temperatures: Vec<f32>,               // fallback schedule (default: [0.0..1.0])
    pub compression_ratio_threshold: f32,      // hallucination filter (default: 2.4)
    pub logprob_threshold: f32,                // low-confidence filter (default: -1.0)
    pub no_speech_threshold: f32,              // silence filter (default: 0.6)
    // ... see src/transcribe.rs for full list
}

pub struct TranscribeResult {
    pub text: String,
    pub segments: Vec<Segment>,
    pub language: String,
}

pub struct Segment {
    pub id: usize,
    pub start: f32,   // seconds
    pub end: f32,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub compression_ratio: f32,
    pub temperature: f32,
}

// Main transcription entry point
pub fn transcribe(
    audio: Array,
    model: &mut Whisper,
    assets_dir: &Path,
    options: &TranscribeOptions,
) -> Result<TranscribeResult>

// Language detection — returns (language_code, {code: probability_map})
pub fn detect_language(
    model: &mut Whisper,
    audio: &Array,
    assets_dir: &Path,
) -> Result<(String, HashMap<String, f32>)>
```

### `decoding` module

```rust
pub struct DecodingOptions {
    pub task: String,
    pub language: Option<String>,
    pub temperature: f32,
    pub sample_len: Option<usize>,
    pub suppress_blank: bool,
    pub suppress_tokens: Option<String>,   // "-1" = use non_speech_tokens
    pub without_timestamps: bool,
    pub max_initial_timestamp: Option<f32>,
    pub prompt: Option<Vec<u32>>,          // previous-context token IDs
}

// Decode a single 30-second mel chunk
pub fn decode(
    model: &mut Whisper,
    mel: &Array,
    tokenizer: &Tokenizer,
    options: &DecodingOptions,
) -> Result<DecodingResult>
```

---

## Supported Models

Any model from [`mlx-community`](https://huggingface.co/mlx-community) in the standard mlx-whisper format.
Common choices:

| Model | HuggingFace repo | Speed | Accuracy |
|-------|-----------------|-------|----------|
| tiny | `mlx-community/whisper-tiny-mlx` | fastest | lowest |
| base | `mlx-community/whisper-base-mlx` | fast | low |
| small | `mlx-community/whisper-small-mlx` | moderate | moderate |
| medium | `mlx-community/whisper-medium-mlx` | moderate | good |
| large-v3 | `mlx-community/whisper-large-v3-mlx` | slow | best |
| large-v3-turbo | `mlx-community/whisper-large-v3-turbo` | fast | best |

Models are downloaded and cached automatically via [`hf-hub`](https://github.com/huggingface/hf-hub).

---

## Project Structure

```
src/
  audio.rs        — load_audio, audio_from_wav_bytes, log_mel_spectrogram, STFT
  whisper.rs      — Encoder/Decoder Transformer architecture (mlx-rs ops)
  load_models.rs  — model weight loading from local path or HuggingFace
  tokenizer.rs    — BPE tokenizer, special tokens, non_speech_tokens
  decoding.rs     — logit filters, KV-cache decode loop, temperature fallback
  transcribe.rs   — sliding-window pipeline, detect_language, transcribe()
  lib.rs          — public re-exports
examples/
  transcribe.rs   — CLI tool
assets/           — *.tiktoken, *.npy (must be copied from mlx_whisper Python package)
```

---

## How It Works

1. **Audio loading** — ffmpeg or in-memory WAV/PCM parsing → float32 mono 16 kHz
2. **Log-Mel spectrogram** — STFT → magnitude² → mel filterbank → log10 → normalise
3. **Encoder** — Audio transformer: Conv1D stem → sinusoidal positional embeddings → N × multi-head self-attention + FFN blocks
4. **Decoder** — Text transformer: token embeddings → positional embeddings → N × masked self-attention + cross-attention (KV cache) + FFN
5. **Logit filters** — SuppressBlank, SuppressTokens, ApplyTimestampRules applied at each step
6. **Token selection** — Greedy (temperature=0) or categorical sampling; accumulate log-prob
7. **Temperature fallback** — if `compression_ratio > 2.4` or `avg_logprob < -1.0`, retry with next temperature
8. **Sliding window** — 30-second chunks with overlap detection via timestamp tokens; seek advances based on decoded segment boundaries

---

## Differences from Python `mlx-whisper`

| Feature | Python | This crate |
|---------|--------|-----------|
| Runtime | Python + mlx | Pure Rust + mlx-rs |
| Word timestamps | ✅ | Not yet |
| VAD filter | ✅ | Not yet |
| `without_timestamps` mode | ✅ | ✅ |
| `translate` task | ✅ | ✅ |
| In-memory audio input | Manual | `audio_from_wav_bytes` / `audio_from_pcm_s16le` |

---

## License

MIT — see [LICENSE](LICENSE).

Based on [mlx-examples/whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) by Apple Inc. (Apache 2.0).
Uses [mlx-rs](https://github.com/oxideai/mlx-rs) Rust bindings for the MLX framework.
