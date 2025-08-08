# server_realtime_v1.py (シンプル版)

import socket
import argparse
import numpy as np
import torch
import torchaudio
import yaml
from omegaconf import DictConfig
import hydra
from modules.commons import recursive_munch, build_model, load_checkpoint
from hf_utils import load_custom_model_from_hf
import threading

# --- ネットワーク設定 ---
HOST = '0.0.0.0'
PORT = 8080
CHUNK_BYTES = 4096 # 2048サンプル * 2バイト/サンプル

# --- グローバル変数 ---
device = None
model_set = {}
reference_cache = {}

def initialize_models(args):
    """V1モデル一式を初期化する"""
    global device, model_set, reference_cache
    print("V1モデル一式を初期化しています...")
    device = torch.device(args.device)
    config = yaml.safe_load(open(args.config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    preprocess_params = recursive_munch(config["preprocess_params"])

    # DiTモデル
    model = build_model(model_params, stage="DiT")
    model, _, _, _ = load_checkpoint(model, None, args.checkpoint_path, load_only_params=True, is_distributed=False)
    for key in model: model[key].eval().to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Vocoder (Hifigan)
    from modules.hifigan.generator import HiFTGenerator
    from modules.hifigan.f0_predictor import ConvRNNF0Predictor
    hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
    vocoder_fn = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
    hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
    vocoder_fn.load_state_dict(torch.load(hift_path, map_location='cpu'))
    vocoder_fn.eval().to(device)

    # Speech Tokenizer (XLSR)
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    tokenizer_config = model_params.speech_tokenizer
    semantic_model_name = tokenizer_config if isinstance(tokenizer_config, str) else tokenizer_config.name
    output_layer = 12 if isinstance(tokenizer_config, str) else tokenizer_config.output_layer
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(semantic_model_name)
    wav2vec_model = Wav2Vec2Model.from_pretrained(semantic_model_name)
    wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
    wav2vec_model.eval().to(device)
    def semantic_fn(waves_16k):
        inputs = wav2vec_feature_extractor(waves_16k.cpu().numpy(), return_tensors="pt", padding=True, sampling_rate=16000).to(device)
        with torch.no_grad(): outputs = wav2vec_model(inputs.input_values)
        return outputs.last_hidden_state.float()

    # Style Encoder (CampPlus)
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=model_params.style_encoder.dim)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval().to(device)
    
    # Mel Spectrogram
    from modules.audio import mel_spectrogram
    # ▼▼▼ configファイルから全てのパラメータを読み込むように修正 ▼▼▼
    mel_fn_args = {
        "n_fft": preprocess_params.spect_params.n_fft,
        "win_size": preprocess_params.spect_params.win_length, # configのwin_lengthをwin_sizeにマッピング
        "hop_size": preprocess_params.spect_params.hop_length,
        "num_mels": preprocess_params.spect_params.n_mels,
        "sampling_rate": preprocess_params.sr,
        "fmin": preprocess_params.spect_params.fmin,
        "fmax": preprocess_params.spect_params.fmax,
        "center": False
    }
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
    
    model_set.update({"model": model, "semantic_fn": semantic_fn, "vocoder_fn": vocoder_fn, "campplus_model": campplus_model, "to_mel": to_mel, "mel_fn_args": mel_fn_args})

    # 目標話者の事前計算
    import librosa
    sr = model_set["mel_fn_args"]["sampling_rate"]
    ref_wave, _ = librosa.load(args.tgtwav, sr=sr)
    ref_wave_tensor = torch.from_numpy(ref_wave).to(device)
    ori_waves_16k = torchaudio.functional.resample(ref_wave_tensor, sr, 16000)
    S_ori = model_set["semantic_fn"](ori_waves_16k.unsqueeze(0))
    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000).to(device)
    style2 = model_set["campplus_model"]((feat2 - feat2.mean(dim=0, keepdim=True)).unsqueeze(0))
    mel2 = model_set["to_mel"](ref_wave_tensor.unsqueeze(0))
    prompt_condition = model_set["model"].length_regulator(S_ori, ylens=torch.LongTensor([mel2.size(2)]).to(device), n_quantizers=3, f0=None)[0]
    reference_cache.update({"prompt_condition": prompt_condition, "mel2": mel2, "style2": style2})
    print("モデルの初期化が完了しました。")

def convert_chunk(chunk_bytes, client_sr, model_sr):
    """単一の音声チャンクを変換する"""
    with torch.no_grad():
        input_chunk_wave = torch.from_numpy(np.frombuffer(chunk_bytes, dtype=np.int16).copy().astype(np.float32) / 32768.0).to(device)
        resampled_chunk = torchaudio.functional.resample(input_chunk_wave, client_sr, model_sr)
        
        input_wav_16k = torchaudio.functional.resample(resampled_chunk, model_sr, 16000)
        S_alt = model_set["semantic_fn"](input_wav_16k.unsqueeze(0))
        target_lengths = torch.LongTensor([S_alt.size(1)]).to(device)
        cond = model_set["model"].length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([reference_cache["prompt_condition"], cond], dim=1)
        
        vc_target = model_set["model"].cfm.inference(
            cat_condition, torch.LongTensor([cat_condition.size(1)]).to(device),
            reference_cache["mel2"], reference_cache["style2"], None, n_timesteps=4
        )
        vc_target = vc_target[:, :, reference_cache["mel2"].size(-1):]
        vc_wave = model_set["vocoder_fn"](vc_target).squeeze()
        
        output_resampled = torchaudio.functional.resample(vc_wave, model_sr, client_sr)
        
        final_output_np = output_resampled.cpu().numpy()
        target_samples = len(input_chunk_wave)
        if len(final_output_np) > target_samples:
            final_output_np = final_output_np[:target_samples]
        elif len(final_output_np) < target_samples:
            padding = np.zeros(target_samples - len(final_output_np), dtype=np.float32)
            final_output_np = np.concatenate((final_output_np, padding))
            
        output_audio_int16 = (final_output_np * 32767.0).astype(np.int16)
        return output_audio_int16.tobytes()

def handle_client_connection(conn, addr, args):
    print(f"\nクライアント接続処理を開始: {addr}")
    try:
        with conn:
            while True:
                data = b''
                while len(data) < CHUNK_BYTES:
                    packet = conn.recv(CHUNK_BYTES - len(data))
                    if not packet: return
                    data += packet
                
                processed_bytes = convert_chunk(data, args.client_sample_rate, model_set['mel_fn_args']['sampling_rate'])
                conn.sendall(processed_bytes)
    except (socket.timeout, ConnectionResetError, ConnectionAbortedError):
        print("クライアントとの接続が切れました。")
    finally:
        print(f"クライアント {addr} との接続処理を終了します。")

def start_server(args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((HOST, PORT)); s.listen();
        print(f"サーバーが {HOST}:{PORT} で待機中です...")
        try:
            while True:
                conn, addr = s.accept()
                threading.Thread(target=handle_client_connection, args=(conn, addr, args), daemon=True).start()
        except KeyboardInterrupt:
            print("\nサーバーをシャットダウンします。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="リアルタイム声質変換サーバー (V1モデル・シンプル版)")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/ITA_for_realtime/DiT_epoch_00234_step_19000.pth", help="V1モデルのチェックポイント")
    parser.add_argument("--config-path", type=str, default="./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml", help="V1モデルの設定ファイル")
    parser.add_argument("--tgtwav", type=str, default="./dataset/ITA-corpus/zundamon/recitation146.wav", help="目標話者のWAVファイル")
    parser.add_argument("--device", type=str, default="cuda", help="使用デバイス (cuda or cpu)")
    parser.add_argument("--client_sample_rate", type=int, default=48000, help="クライアントが使用するサンプリングレート")
    args = parser.parse_args()
    try:
        initialize_models(args)
        start_server(args)
    except Exception as e:
        print(f"起動中にエラーが発生しました: {e}")
    print("サーバープログラムを終了します。")