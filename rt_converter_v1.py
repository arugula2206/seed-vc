# rt_converter_v1.py (V1リアルタイムconfig対応版)

import torch
import yaml
import numpy as np
import torchaudio
from omegaconf import DictConfig
import hydra
from accelerate import Accelerator
import librosa
from modules.commons import recursive_munch, build_model, load_checkpoint
from hf_utils import load_custom_model_from_hf

# --- グローバル変数 ---
device = None
model_set = {}
reference_cache = {}
audio_buffer = None
buffer_config = {}

def initialize_models(args):
    """
    V1リアルタイム用設定ファイルに基づいてモデル一式を初期化する
    """
    global device, model_set, reference_cache

    print("V1モデル一式を初期化しています...")
    device = torch.device(args.device)

    try:
        # --- 設定ファイルの読み込み ---
        print("設定ファイルを読み込んでいます...")
        config = yaml.safe_load(open(args.config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        preprocess_params = recursive_munch(config["preprocess_params"])
        model_params.dit_type = 'DiT'
        
        print("DiTモデルを構築・読み込みしています...")
        model = build_model(model_params, stage="DiT")
        model, _, _, _ = load_checkpoint(
            model, None, args.checkpoint_path, load_only_params=True, is_distributed=False
        )
        for key in model:
            model[key].eval().to(device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # --- 各種エンコーダー、デコーダーの読み込み ---
        print("各種エンコーダー/ボコーダーを読み込んでいます...")
        
        # Vocoderの読み込み (configに基づきHifiganをロード)
        if model_params.vocoder.type == 'hifigan':
            print("Hifigan vocoderを読み込んでいます...")
            from modules.hifigan.generator import HiFTGenerator
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
            vocoder_fn = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            vocoder_fn.load_state_dict(torch.load(hift_path, map_location='cpu'))
            vocoder_fn.eval().to(device)
        else:
            raise ValueError(f"設定ファイルで指定されたVocoderタイプ '{model_params.vocoder.type}' は現在サポートされていません。")

        # Speech Tokenizerの読み込み (configに基づきXLSRをロード)
        if model_params.speech_tokenizer.type == 'xlsr':
            print("XLSR speech tokenizerを読み込んでいます...")
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            semantic_model_name = model_params.speech_tokenizer.name
            output_layer = model_params.speech_tokenizer.output_layer
            wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(semantic_model_name)
            wav2vec_model = Wav2Vec2Model.from_pretrained(semantic_model_name)
            wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
            wav2vec_model.eval().to(device)
            def semantic_fn(waves_16k):
                inputs = wav2vec_feature_extractor(waves_16k.cpu().numpy(), return_tensors="pt", padding=True, sampling_rate=16000).to(device)
                with torch.no_grad():
                    outputs = wav2vec_model(inputs.input_values)
                return outputs.last_hidden_state.float()
        else:
            raise ValueError(f"設定ファイルで指定されたTokenizerタイプ '{model_params.speech_tokenizer.type}' は現在サポートされていません。")

        # Style Encoder (CampPlus) の読み込み
        print("CampPlus style encoderを読み込んでいます...")
        from modules.campplus.DTDNN import CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=model_params.style_encoder.dim)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval().to(device)

        # Mel Spectrogram計算関数の準備
        from modules.audio import mel_spectrogram
        # パスをpreprocess_params.spect_paramsに変更
        mel_fn_args = {
            "n_fft": preprocess_params.spect_params.n_fft,
            "win_size": preprocess_params.spect_params.win_length,
            "hop_size": preprocess_params.spect_params.hop_length,
            "num_mels": preprocess_params.spect_params.n_mels,
            "sampling_rate": preprocess_params.sr,
            "fmin": preprocess_params.spect_params.fmin,
            "fmax": preprocess_params.spect_params.fmax,
            "center": False
        }
        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        model_set.update({
            "model": model, "semantic_fn": semantic_fn, "vocoder_fn": vocoder_fn,
            "campplus_model": campplus_model, "to_mel": to_mel, "mel_fn_args": mel_fn_args
        })
        
        # --- 目標話者の事前計算 ---
        print("目標話者のデータを事前計算しています...")
        sr = model_set["mel_fn_args"]["sampling_rate"]
        ref_wave, _ = librosa.load(args.tgtwav, sr=sr)
        ref_wave_tensor = torch.from_numpy(ref_wave).to(device)
        
        ori_waves_16k = torchaudio.functional.resample(ref_wave_tensor, sr, 16000)
        S_ori = model_set["semantic_fn"](ori_waves_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000).to(device)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = model_set["campplus_model"](feat2.unsqueeze(0))
        mel2 = model_set["to_mel"](ref_wave_tensor.unsqueeze(0))
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        
        reference_cache.update({
            "prompt_condition": model_set["model"].length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)[0],
            "mel2": mel2, 
            "style2": style2
        })
        print("モデルの初期化が完了しました。")
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}")
        raise

def setup_audio_buffer(chunk_samples, client_sr, model_sr):
    global audio_buffer, buffer_config
    extra_time = 2.0
    extra_frame = int(extra_time * model_sr)
    buffer_config = { 'chunk_samples': chunk_samples, 'model_sr': model_sr, 'client_sr': client_sr }
    total_buffer_size = extra_frame + chunk_samples
    audio_buffer = torch.zeros(total_buffer_size, device=device, dtype=torch.float32)
    print(f"オーディオバッファを初期化しました (サイズ: {total_buffer_size} サンプル)")

def convert_chunk(chunk_bytes, client_sr=48000, diffusion_steps=4):
    global audio_buffer
    if not model_set: raise RuntimeError("モデルが初期化されていません。")

    with torch.no_grad():
        input_chunk_wave = torch.from_numpy(np.frombuffer(chunk_bytes, dtype=np.int16).copy().astype(np.float32) / 32768.0).to(device)
        resampled_chunk = torchaudio.functional.resample(input_chunk_wave, client_sr, buffer_config['model_sr'])
        
        # ローリングバッファを更新 (リサンプリング後の正しいサンプル数で)
        num_new_samples = resampled_chunk.shape[0]
        audio_buffer = torch.roll(audio_buffer, -num_new_samples, dims=0)
        audio_buffer[-num_new_samples:] = resampled_chunk

        input_wav_16k = torchaudio.functional.resample(audio_buffer, buffer_config['model_sr'], 16000)
        S_alt = model_set["semantic_fn"](input_wav_16k.unsqueeze(0))
        target_lengths = torch.LongTensor([S_alt.size(1)]).to(device)
        cond = model_set["model"].length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([reference_cache["prompt_condition"], cond], dim=1)
        
        vc_target = model_set["model"].cfm.inference(
            cat_condition, torch.LongTensor([cat_condition.size(1)]).to(device),
            reference_cache["mel2"], reference_cache["style2"], None, n_timesteps=diffusion_steps
        )
        vc_target = vc_target[:, :, reference_cache["mel2"].size(-1):]
        vc_wave = model_set["vocoder_fn"](vc_target).squeeze()
        
        output_chunk = vc_wave[-buffer_config['chunk_samples']:]

        # --- 出力音声の準備 ---
        output_resampled = torchaudio.functional.resample(output_chunk, buffer_config['model_sr'], client_sr)
        
        # ▼▼▼ サイズを強制的に入力チャンクのサンプル数に合わせる処理を追加 ▼▼▼
        final_output_np = output_resampled.cpu().numpy()
        target_samples = buffer_config['chunk_samples']

        if len(final_output_np) > target_samples:
            # もし長ければ切り捨てる
            final_output_np = final_output_np[:target_samples]
        elif len(final_output_np) < target_samples:
            # もし短ければ無音で埋める
            padding = np.zeros(target_samples - len(final_output_np), dtype=np.float32)
            final_output_np = np.concatenate((final_output_np, padding))
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        output_audio_int16 = (final_output_np * 32767.0).astype(np.int16)
        return output_audio_int16.tobytes()