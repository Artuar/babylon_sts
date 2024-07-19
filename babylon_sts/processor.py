import os
import numpy as np
import whisper_timestamped as whisper
import torch
from pydub import AudioSegment
from datetime import datetime
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Dict, Tuple, Optional

lang_settings = {
    'ua': {
        'translation_key': 'uk',
        'speaker': 'v4_ua',
        'speaker_name': 'mykyta'
    },
    'ru': {
        'translation_key': 'ru',
        'speaker': 'v4_ru',
        'speaker_name': 'aidar'
    },
    'fr': {
        'translation_key': 'fr',
        'speaker': 'v3_fr',
        'speaker_name': 'fr_0'
    },
    'de': {
        'translation_key': 'de',
        'speaker': 'v3_de',
        'speaker_name': 'karlsson'
    },
    'es': {
        'translation_key': 'es',
        'speaker': 'v3_es',
        'speaker_name': 'es_0'
    },
    'en': {
        'translation_key': 'en',
        'speaker': 'v3_en',
        'speaker_name': 'en_0'
    }
}


def load_or_download_translation_model(language: str) -> Tuple[MarianTokenizer, MarianMTModel]:
    """
    Load or download the translation model for the specified language.

    Args:
        language (str): The language code. Possible values: 'en', 'ua', 'ru', 'fr', 'de', 'es'.

    Returns:
        Tuple[MarianTokenizer, MarianMTModel]: The tokenizer and translation model.
    """
    model_name = f"Helsinki-NLP/opus-mt-en-{lang_settings[language]['translation_key']}"
    local_dir = f"local_model_{language}"
    if os.path.exists(local_dir):
        tokenizer = MarianTokenizer.from_pretrained(local_dir)
        translation_model = MarianMTModel.from_pretrained(local_dir)
    else:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        translation_model = MarianMTModel.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        translation_model.save_pretrained(local_dir)
    return tokenizer, translation_model


def load_silero_model(language: str) -> torch.nn.Module:
    """
    Load the Silero TTS model for the specified language.

    Args:
        language (str): The language code. Possible values: 'en', 'ua', 'ru', 'fr', 'de', 'es'.

    Returns:
        torch.nn.Module: The TTS model.
    """
    return torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=language, speaker=lang_settings[language]['speaker'])


class AudioProcessor:
    def __init__(self, language: str, model_name: str, sample_rate: int = 24000):
        """
        Initialize the AudioProcessor with the specified language, Whisper model, and sample rate.

        Args:
            language (str): The language code. Possible values: 'en', 'ua', 'ru', 'fr', 'de', 'es'.
            model_name (str): The Whisper model to use. Possible values: 'tiny', 'base', 'small', 'medium', 'large'.
            sample_rate (int): The sample rate for audio processing.
        """
        self.language = language
        self.sample_rate = sample_rate
        self.audio_model = whisper.load_model(model_name)
        self.tokenizer, self.translation_model = load_or_download_translation_model(language)
        self.tts_model, self.example_text = load_silero_model(language)
        self.tts_model.to(torch.device('cpu'))

    def translate_text(self, text: str) -> str:
        """
        Translate the given text to the target language.

        Args:
            text (str): The text to translate.

        Returns:
            str: The translated text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        translated = self.translation_model.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translated_text[0]

    def synthesize_speech(self, text: str) -> np.ndarray:
        """
        Synthesize speech from the given text.

        Args:
            text (str): The text to synthesize.

        Returns:
            np.ndarray: The synthesized speech audio.
        """
        audio = self.tts_model.apply_tts(text=text, sample_rate=self.sample_rate, speaker=lang_settings[self.language]['speaker_name'])
        return audio

    def recognize_speech(self, audio_data: bytes) -> List[Dict[str, str]]:
        """
        Recognize speech from the given audio data.

        Args:
            audio_data (bytes): The audio data to recognize.

        Returns:
            List[Dict[str, str]]: The recognized segments with text.
        """
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=2,
            frame_rate=16000,
            channels=1
        )
        audio_segment = audio_segment.normalize()
        samples = np.array(audio_segment.get_array_of_samples())
        audio_np = samples.astype(np.float32) / 32768.0

        result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        return result['segments']

    def process_audio(self, timestamp: datetime, audio_data: bytes) -> Tuple[np.ndarray, Optional[Dict[str, str]]]:
        """
        Process the audio data by recognizing speech, translating text, and synthesizing speech.

        Args:
            timestamp (datetime): The timestamp of the audio data.
            audio_data (bytes): The audio data to process.

        Returns:
            Tuple[np.ndarray, Optional[Dict[str, str]]]: The final audio and log data.
        """
        segments = self.recognize_speech(audio_data)

        if not segments:
            return np.array(audio_data), {
                "timestamp": timestamp,
                "original_text": '',
                "translated_text": '',
                "synthesis_delay": 0
            }

        translated_segments = []
        for segment in segments:
            translated_text = self.translate_text(segment['text'])
            translated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': translated_text
            })

        final_audio = np.array([])
        for segment in translated_segments:
            synthesized_segment = self.synthesize_speech(segment['text'])
            silence_duration = int(segment['start'] * self.sample_rate) - len(final_audio)
            if silence_duration > 0:
                final_audio = np.pad(final_audio, (0, silence_duration), 'constant')
            final_audio = np.concatenate((final_audio, synthesized_segment), axis=None)

        synthesis_timestamp = datetime.utcnow()
        synthesis_delay = (synthesis_timestamp - timestamp).total_seconds()

        log_data = {
            "timestamp": timestamp,
            "original_text": segments[-1]['text'],
            "translated_text": translated_segments[-1]['text'],
            "synthesis_delay": synthesis_delay
        }

        return final_audio, log_data
