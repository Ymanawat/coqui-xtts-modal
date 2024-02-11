import base64
import io
import os
from shutil import rmtree
import tempfile
import time
import wave
import torch
import numpy as np
from typing import List
from pydantic import BaseModel
from modal import Image, asgi_app
from fastapi import FastAPI, UploadFile, requests
from fastapi.responses import StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import modal

prev_log_time = None

def download_model():
    device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.") 

    custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")

    # Check if the model directory exists and has the required config file
    if os.path.exists(custom_model_path) and os.path.isfile(os.path.join(custom_model_path, "config.json")):
        print("Loading custom model from", model_path, flush=True)
    else:
        print("Loading default model", flush=True)
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print("Downloading XTTS Model:", model_name, flush=True)

        manager = ModelManager()  # Initialize the ModelManager
        model_item, model_full_name, model, md5sum = manager._set_model_item(model_name)
        output_path = os.path.join(manager.output_prefix, model_full_name)

        # Check if the model is already downloaded
        if os.path.exists(output_path) and os.path.isfile(os.path.join(output_path, "config.json")):
            model_path = output_path
            print("XTTS Model already downloaded", flush=True)
        else:
            os.makedirs(output_path, exist_ok=True)
            print(f" > Downloading model to {output_path}")

            try:
                if "fairseq" in model_name:
                    manager.download_fairseq_model(model_name, output_path)
                elif "github_rls_url" in model_item:
                    manager._download_github_model(model_item, output_path)
                elif "hf_url" in model_item:
                    manager._download_hf_model(model_item, output_path)

            except requests.RequestException as e:
                print(f" > Failed to download the model file to {output_path}")
                rmtree(output_path)
                raise e

            manager.print_model_license(model_item=model_item)

            output_model_path, output_config_path = manager._find_files(output_path)
            manager._update_paths(output_path, output_config_path)

            model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
            print("XTTS Model downloaded", flush=True)

            print(model_path)

    return model_path


def load_model(model_path='/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2'):
    import torch 
    device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.") 

    print("Loading XTTS", flush=True)
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)
    model.to(device)
    print("XTTS Loaded.", flush=True)
    print("Running XTTS Server ...", flush=True)

    return model, config

def log(message: str) -> None:
    current_time = time.perf_counter()
    if hasattr(log, 'prev_log_time') and log.prev_log_time is not None:
        time_diff = current_time - log.prev_log_time
        print(f"[LOG] {message} - Time elapsed since previous log: {time_diff:.5f}s")
    else:
        print(f"[LOG] {message}")
    log.prev_log_time = current_time

gpu_image = (
    Image.debian_slim()
    .apt_install('git')
    .pip_install('TTS @ git+https://github.com/coqui-ai/TTS@fa28f99f1508b5b5366539b2149963edcb80ba62', 'uvicorn[standard]==0.23.2', 'fastapi==0.95.2', 'deepspeed==0.10.3', 'pydantic==1.10.13', 'python-multipart==0.0.6', 'typing-extensions>=4.8.0', 'numpy==1.24.3', 'cutlet', 'mecab-python3==1.0.6', 'unidic-lite==1.0.8', 'unidic==1.1.0')
    .run_function(download_model)
)

stub = modal.Stub('xtts-streaming-2', image= gpu_image)

@stub.function(
    image= gpu_image,
    container_idle_timeout=300,
    timeout=240,
    gpu='T4'
)
@asgi_app()
def coqui_app():
    model, config = load_model()

    app = FastAPI(
        title="XTTS Streaming server",
        description="""XTTS Streaming server""",
        version="0.0.1",
        docs_url="/",
        port=8000
    )

    class StreamingInputs(BaseModel):
        speaker_embedding: List[float]
        gpt_cond_latent: List[List[float]]
        text: str
        language: str
        add_wav_header: bool = True
        stream_chunk_size: str = "20"

    @app.get("/hi")
    async def hello():
        return 'hello'

    def postprocess(self, wav):  # Add 'self' parameter
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = wav[None, : int(wav.shape[0])]
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def encode_audio_common(self, frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1):  # Add 'self' parameter
        """Return base64 encoded audio"""
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(channels)
            vfout.setsampwidth(sample_width)
            vfout.setframerate(sample_rate)
            vfout.writeframes(frame_input)

        wav_buf.seek(0)
        if encode_base64:
            b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
            return b64_encoded
        else:
            return wav_buf.read()

    @app.post("/clone_speaker")
    async def predict_speaker(self, wav_file: UploadFile):
        """Compute conditioning inputs from reference audio file."""
        temp_audio_name = next(tempfile._get_candidate_names())
        with open(temp_audio_name, "wb") as temp, torch.inference_mode():
            temp.write(io.BytesIO(wav_file.file.read()).getbuffer())
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                temp_audio_name
            )
        return {
            "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
            "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
        }

    @app.post("/tts_stream")
    async def predict_streaming_endpoint(self, parsed_input):
        log('before inference')
        speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
        text = parsed_input.text
        language = parsed_input.language
        stream_chunk_size = int(parsed_input.stream_chunk_size)
        add_wav_header = parsed_input.add_wav_header

        chunks = model.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=stream_chunk_size,
                enable_text_splitting=True,
            )
        
        def audio_chunks():
            for i, chunk in enumerate(chunks):
                chunk = postprocess(chunk)
                if i == 0 and add_wav_header:
                    yield encode_audio_common(b"", encode_base64=False)
                    yield chunk.tobytes()
                    log('first bytes sent')
                else:
                    yield chunk.tobytes()

        return StreamingResponse(audio_chunks(), media_type="audio/wav")

    class TTSInputs(BaseModel):
        speaker_embedding: List[float]
        gpt_cond_latent: List[List[float]]
        text: str
        language: str

    @app.post("/tts")
    async def predict_speech(self, parsed_input):
        speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
        text = parsed_input.text
        language = parsed_input.language

        out = model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
        )
        
        wav = postprocess(wav = torch.tensor(out["wav"]))
        return encode_audio_common(frame_input = wav.tobytes())


    @app.get("/studio_speakers")
    async def get_speakers(self):
        if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
            return {
                speaker: {
                    "speaker_embedding": model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                    "gpt_cond_latent": model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
                }
                for speaker in model.speaker_manager.speakers.keys()
            }
        else:
            return {}
            
    @app.get("/languages")
    async def get_languages(self):
        return config.languages

    return app
