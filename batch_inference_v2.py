import os
import argparse
import torch
import yaml
import soundfile as sf
import time
from modules.commons import str2bool

import glob

# Set up device and torch configurations
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

# Global variables to store model instances
vc_wrapper_v2 = None


def load_v2_models(args):
    """Load V2 models using the wrapper from app.py"""
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper


def convert_voice_v2(source_audio_path, target_audio_path, args):
    """Convert voice using V2 model"""
    global vc_wrapper_v2
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)

    # Use the generator function but collect all outputs
    generator = vc_wrapper_v2.convert_voice_with_streaming(
        source_audio_path=source_audio_path,
        target_audio_path=target_audio_path,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligebility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        convert_style=args.convert_style,
        anonymization_only=args.anonymization_only,
        device=device,
        dtype=dtype,
        stream_output=True
    )

    # Collect all outputs from the generator
    for output in generator:
        _, full_audio = output
    return full_audio


def main(args):
    # --- 1. 話者ディレクトリを自動でリストアップ ---
    parent_target_dir = args.target
    if not os.path.isdir(parent_target_dir):
        print(f"エラー: --target には話者ディレクトリが含まれる親ディレクトリを指定してください。例: ./dataset/ITA-corpus")
        return

    try:
        speaker_dirs = [d.path for d in os.scandir(parent_target_dir) if d.is_dir()]
    except FileNotFoundError:
        print(f"エラー: ディレクトリが見つかりません: {parent_target_dir}")
        return

    if not speaker_dirs:
        print(f"エラー: {parent_target_dir} 内に話者のサブディレクトリが見つかりません。")
        return

    print(f"{len(speaker_dirs)} 人の話者を検出しました。一括処理を開始します...")
    print("モデルを読み込みます（この処理は一度だけです）...")

    # --- 2. 話者ごとにループ ---
    for speaker_dir in speaker_dirs:
        speaker_name = os.path.basename(speaker_dir)
        
        # この話者用の出力ディレクトリパスを定義
        speaker_output_dir = os.path.join(args.output, speaker_name)
        
        # 出力ディレクトリが既に存在する場合、この話者の処理をスキップ
        if os.path.isdir(speaker_output_dir):
            print(f"\n話者 '{speaker_name}' の出力フォルダは既に存在するため、スキップします。")
            continue # 次の話者へ

        print(f"\n=============================================")
        print(f"話者: {speaker_name} の処理を開始")
        print(f"=============================================")

        # 出力ディレクトリを作成
        os.makedirs(speaker_output_dir, exist_ok=True)

        # 話者ディレクトリ内の.wavファイルをすべて取得
        target_files = glob.glob(os.path.join(speaker_dir, '*.wav'))
        if not target_files:
            print(f"{speaker_name} のディレクトリに .wav ファイルが見つからないため、スキップします。")
            continue

        # 参照音声ごとにループ
        for target_path in target_files:
            print(f"--- 参照音声: {os.path.basename(target_path)} を処理中...")
            start_time = time.time()
            
            converted_audio = convert_voice_v2(args.source, target_path, args)
            end_time = time.time()

            if converted_audio is None:
                print("エラー: 音声変換に失敗しました。")
                continue

            # --- 3. ファイルを保存 ---
            model_identifier = "default"
            if args.cfm_checkpoint_path:
                model_identifier = os.path.splitext(os.path.basename(args.cfm_checkpoint_path))[0]
            
            source_name = os.path.splitext(os.path.basename(args.source))[0]
            target_wav_name = os.path.splitext(os.path.basename(target_path))[0]
            filename = f"vc_v2_{source_name}_to_{target_wav_name}_by_{model_identifier}.wav"
            output_path = os.path.join(speaker_output_dir, filename)

            save_sr, audio_data = converted_audio
            sf.write(output_path, audio_data, save_sr)
            print(f"変換完了 ({end_time - start_time:.2f}秒), 保存先: {output_path}")

    print("\nすべての話者の処理が完了しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Conversion Inference Script")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source audio file")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target/reference audio file OR a directory containing .wav files")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory for converted audio")
    parser.add_argument("--diffusion-steps", type=int, default=30,
                        help="Number of diffusion steps")
    parser.add_argument("--length-adjust", type=float, default=1.0,
                        help="Length adjustment factor (<1.0 for speed-up, >1.0 for slow-down)")
    parser.add_argument("--compile", type=bool, default=False,
                        help="Whether to compile the model for faster inference")

    # V2 specific arguments
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7,
                        help="Intelligibility CFG rate for V2 model")
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7,
                        help="Similarity CFG rate for V2 model")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter for V2 model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature sampling parameter for V2 model")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for V2 model")
    parser.add_argument("--convert-style", type=str2bool, default=False,
                        help="Convert style/emotion/accent for V2 model")
    parser.add_argument("--anonymization-only", type=str2bool, default=False,
                        help="Anonymization only mode for V2 model")

    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")

    args = parser.parse_args()
    main(args)