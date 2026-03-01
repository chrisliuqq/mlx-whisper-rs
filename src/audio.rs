// Translated from mlx-whisper/audio.py (Apple Inc.)

use std::f32::consts::PI;

use anyhow::{bail, Result};
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
use mlx_rs::ops::{as_strided, concatenate, maximum, pad};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

// Hard-coded audio hyperparameters
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000
pub const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2;
pub const FRAMES_PER_SECOND: usize = SAMPLE_RATE / HOP_LENGTH;
pub const TOKENS_PER_SECOND: usize = SAMPLE_RATE / N_SAMPLES_PER_TOKEN;

/// Load audio from file path via ffmpeg, returning float32 mono 16kHz samples.
/// 對應 Python: load_audio()
pub fn load_audio(file: &str, sr: usize) -> Result<Array> {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-nostdin", "-i", file,
            "-threads", "0",
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", &sr.to_string(),
            "-",
        ])
        .output()?;

    if !output.status.success() {
        bail!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let samples: Vec<f32> = output.stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();

    Ok(Array::from_slice(&samples, &[samples.len() as i32]))
}

/// Trim or pad `array` to exactly `length` elements along axis 0.
///
/// Works for both 1-D audio `[n_samples]` and 2-D mel `[n_frames, n_mels]`.
/// 對應 Python: pad_or_trim()
pub fn pad_or_trim(array: Array, length: usize) -> Result<Array> {
    let len = length as i32;
    let current = array.shape()[0];

    let array = if current > len {
        array.index(..len)
    } else {
        array
    };

    let current = array.shape()[0];
    if current < len {
        let pad_amount = (len - current) as i32;
        let ndim = array.ndim();
        let mut widths = vec![(0i32, 0i32); ndim];
        widths[0] = (0, pad_amount); // pad axis 0 (samples / frames)
        Ok(pad(&array, widths.as_slice(), None, None)?)
    } else {
        Ok(array)
    }
}

/// Hanning window of given size（periodic）
/// 對應 Python: hanning(size) = np.hanning(size + 1)[:-1]
pub fn hanning(size: usize) -> Array {
    let values: Vec<f32> = (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect();
    Array::from_slice(&values, &[size as i32])
}

/// Mel filterbank matrix, shape (n_mels, N_FFT/2 + 1).
/// 從 .npy 檔案讀取（由 assets/mel_filters_80.npy 等提供）
/// 對應 Python: mel_filters(n_mels)
pub fn mel_filters(n_mels: usize, assets_dir: &std::path::Path) -> Result<Array> {
    assert!(n_mels == 80 || n_mels == 128, "n_mels must be 80 or 128");
    let path = assets_dir.join(format!("mel_filters_{n_mels}.npy"));
    load_npy_f32(&path, &[n_mels as i32, (N_FFT / 2 + 1) as i32])
}

/// 最小化 .npy float32 reader（不依賴外部 crate）
fn load_npy_f32(path: &std::path::Path, expected_shape: &[i32]) -> Result<Array> {
    let bytes = std::fs::read(path)?;

    // Magic: \x93NUMPY
    if &bytes[..6] != b"\x93NUMPY" {
        bail!("Not a valid .npy file: {:?}", path);
    }

    // Header length at bytes 8-10 (little-endian u16)
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_start = 10 + header_len;

    // Parse float32 data
    let data: Vec<f32> = bytes[data_start..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(Array::from_slice(&data, expected_shape))
}

/// Short-time Fourier Transform
/// 對應 Python: stft()
fn stft(x: &Array, window: &Array, nperseg: usize, noverlap: usize, nfft: usize) -> Result<Array> {
    let padding = (nperseg / 2) as i32;

    // reflect padding：prefix = x[1:padding+1][::-1], suffix = x[-(padding+1):-1][::-1]
    let prefix = x.index(1..(padding + 1));
    let prefix = prefix.index((..prefix.shape()[0]).stride_by(-1));

    let suffix = x.index(-(padding + 1)..-1);
    let suffix = suffix.index((..suffix.shape()[0]).stride_by(-1));

    let x = concatenate(&[prefix, x.clone(), suffix])?;

    // sliding window via as_strided
    let t = (x.size() - nperseg + noverlap) / noverlap;
    let shape = [t as i32, nfft as i32];
    let strides = [noverlap as i64, 1i64];
    let x = as_strided(&x, &shape[..], &strides[..], None)?;

    // windowed rfft
    let windowed = &x * window;
    let spectrum = mlx_rs::fft::rfft(&windowed, None, None)?;
    Ok(spectrum)
}

/// Compute log-Mel spectrogram from float32 audio samples.
/// 對應 Python: log_mel_spectrogram()
///
/// - `audio`: 16kHz mono float32 samples
/// - `n_mels`: 80 or 128
/// - `assets_dir`: 包含 mel_filters_{n_mels}.npy 的目錄
pub fn log_mel_spectrogram(
    audio: Array,
    n_mels: usize,
    assets_dir: &std::path::Path,
) -> Result<Array> {
    let window = hanning(N_FFT);
    let freqs = stft(&audio, &window, N_FFT, HOP_LENGTH, N_FFT)?;

    // magnitudes = freqs[:-1, :].abs().square()
    let n_frames = freqs.shape()[0] - 1;
    let magnitudes = freqs.index(..n_frames).abs()?.square()?;

    // mel_spec = magnitudes @ filters.T
    let filters = mel_filters(n_mels, assets_dir)?;
    let filters_t = mlx_rs::ops::transpose(&filters)?;
    let mel_spec = mlx_rs::ops::matmul(&magnitudes, &filters_t)?;

    // log_spec = maximum(mel_spec, 1e-10).log10()
    let floor = Array::from_f32(1e-10_f32);
    let log_spec = maximum(&mel_spec, &floor)?.log10()?;

    // log_spec = maximum(log_spec, log_spec.max() - 8.0)
    let max_val = log_spec.max(None)? - 8.0_f32;
    let log_spec = maximum(&log_spec, &max_val)?;

    // log_spec = (log_spec + 4.0) / 4.0
    let log_spec = (log_spec + 4.0_f32) / 4.0_f32;

    eval([&log_spec])?;
    Ok(log_spec)
}

// ── In-memory audio helpers ───────────────────────────────────────────────────

/// Parse WAV bytes (RIFF/PCM) from memory and return float32 mono samples.
///
/// Supports PCM s16le (most recorders), PCM s32le, and IEEE float32.
/// Multi-channel audio is averaged to mono.
///
/// # Returns
/// `(samples, sample_rate)` — caller must ensure `sample_rate == SAMPLE_RATE`
/// (16000) before passing to [`log_mel_spectrogram`].
pub fn audio_from_wav_bytes(bytes: &[u8]) -> Result<(Array, u32)> {
    // RIFF/WAVE header
    if bytes.len() < 12 {
        bail!("WAV too short");
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("Not a valid WAV file (missing RIFF/WAVE header)");
    }

    let u16le = |off: usize| u16::from_le_bytes([bytes[off], bytes[off + 1]]);
    let u32le = |off: usize| u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);

    // Scan chunks after RIFF header
    let mut pos = 12usize;
    let mut audio_format = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_start = 0usize;
    let mut data_size = 0usize;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32le(pos + 4) as usize;
        let data_off = pos + 8;

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                bail!("fmt chunk too small");
            }
            audio_format   = u16le(data_off);        // 1 = PCM, 3 = IEEE float
            channels       = u16le(data_off + 2);
            sample_rate    = u32le(data_off + 4);
            bits_per_sample = u16le(data_off + 14);
        } else if chunk_id == b"data" {
            data_start = data_off;
            data_size  = chunk_size.min(bytes.len().saturating_sub(data_off));
        }

        // Chunks are word-aligned
        pos = data_off + ((chunk_size + 1) & !1);
    }

    if channels == 0 || data_start == 0 {
        bail!("WAV missing fmt or data chunk");
    }

    let raw = &bytes[data_start..data_start + data_size];

    // Convert to f32 interleaved samples
    let interleaved: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => raw
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
        (1, 32) => raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0)
            .collect(),
        (3, 32) => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => bail!(
            "Unsupported WAV format: audio_format={audio_format}, bits_per_sample={bits_per_sample}"
        ),
    };

    // Downmix to mono
    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        let ch = channels as usize;
        let n_frames = interleaved.len() / ch;
        (0..n_frames)
            .map(|i| interleaved[i * ch..(i + 1) * ch].iter().sum::<f32>() / ch as f32)
            .collect()
    };

    Ok((Array::from_slice(&mono, &[mono.len() as i32]), sample_rate))
}

/// Convert raw PCM s16le bytes (already mono) to a normalised float32 Array.
///
/// Use this when you have raw microphone samples without a WAV header.
/// Assumes the data is already at the target sample rate (16 kHz for Whisper).
pub fn audio_from_pcm_s16le(pcm_bytes: &[u8]) -> Array {
    let samples: Vec<f32> = pcm_bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    Array::from_slice(&samples, &[samples.len() as i32])
}
