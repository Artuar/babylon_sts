import os

import numpy as np
import whisper_timestamped as whisper
import torch
from pydub import AudioSegment
from datetime import datetime
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Dict, Tuple, Optional, TypedDict

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
    },
    'hi': {
        'translation_key': 'hi',
        'speaker': 'v4_indic',
        'speaker_name': 'hindi_male'
    }
}


class RecognizeResult(TypedDict):
    text: str
    segments: List[Dict[str, str]]
    language: str


def load_or_download_translation_model(language_to: str, language_from: str) -> Tuple[MarianTokenizer, MarianMTModel]:
    model_name = f"Helsinki-NLP/opus-mt-{lang_settings[language_from]['translation_key']}-{lang_settings[language_to]['translation_key']}"
    local_dir = f"local_model_{language_from}_{language_to}"
    try:
        if os.path.exists(local_dir):
            tokenizer = MarianTokenizer.from_pretrained(local_dir)
            translation_model = MarianMTModel.from_pretrained(local_dir)
        else:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            translation_model = MarianMTModel.from_pretrained(model_name)
            tokenizer.save_pretrained(local_dir)
            translation_model.save_pretrained(local_dir)
        return tokenizer, translation_model
    except Exception as e:
        raise ValueError(f"Error loading translation model for {language_from} to {language_to}: {e}")


def load_silero_model(language: str, speaker: str) -> torch.nn.Module:
    return torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=speaker
    )


class AudioProcessor:
    def __init__(
            self,
            language_to: str,
            language_from: str,
            model_name: str,
            speaker: Optional[str] = None,
            speaker_name: Optional[str] = None,
            sample_rate: int = 24000
    ):
        """
        Initialize the AudioProcessor with the specified language, Whisper model, and sample rate.

        Args:
            language_to (str): The language code. Possible values: 'en', 'ua', 'ru', 'fr', 'de', 'es', 'hi'.
            language_from (str): The language code. Possible values: 'en', 'ua', 'ru', 'fr', 'de', 'es', 'hi'.
            model_name (str): The Whisper model to use. Possible values: 'tiny', 'base', 'small', 'medium', 'large'.
            sample_rate (int): The sample rate for audio processing.
            speaker (Optional[str]): The key of Silero model speaker for speech synthesize.
            speaker_name (Optional[str]): The name of Silero model speaker_name for speech synthesize.
        """
        self.language_to = language_to
        self.language_from = language_from
        self.sample_rate = sample_rate
        self.speaker = speaker or lang_settings[language_to]['speaker']
        self.speaker_name = speaker_name or lang_settings[language_to]['speaker_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.audio_model = whisper.load_model(model_name)
        self.tokenizer, self.translation_model = load_or_download_translation_model(language_to, language_from)
        self.tts_model, self.example_text = load_silero_model(language_to, self.speaker)

        self.audio_model.to(self.device)
        self.translation_model.to(self.device)
        self.tts_model.to(self.device)

    def normalize_audio(self, audio_data: bytes) -> Tuple[np.ndarray, float]:
        """
        Normalize the given audio data.

        Args:
            audio_data (bytes): The audio data to normalize.

        Returns:
            Tuple[np.ndarray, float]: The normalized audio data and length in seconds.
        """
        try:
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=self.sample_rate,
                channels=1
            )
            audio_segment = audio_segment.normalize()
            samples = np.array(audio_segment.get_array_of_samples())
            audio_np = samples.astype(np.float32) / 32768.0
            return audio_np, len(audio_segment) / 1000
        except Exception as e:
            raise ValueError(f"Normalize audio error: {e}")

    def adjust_audio_length(self, audio_np: np.ndarray, target_length: int) -> np.ndarray:
        current_length = len(audio_np)
        if current_length < target_length:
            padding_length = target_length - current_length
            padded_audio_np = np.pad(audio_np, (0, padding_length), 'constant')
            return padded_audio_np
        else:
            return audio_np

    def translate_text(self, text: str) -> str:
        """
        Translate the given text to the target language.

        Args:
            text (str): The text to translate.

        Returns:
            str: The translated text.
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            translated = self.translation_model.generate(**inputs)
            translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            return translated_text[0]
        except Exception as e:
            raise ValueError(f"Translated error '{text}': {e}")

    def synthesize_speech(self, text: str) -> np.ndarray:
        """
        Synthesize speech from the given text.

        Args:
            text (str): The text to synthesize.

        Returns:
            np.ndarray: The synthesized speech audio.
        """
        try:
            audio = self.tts_model.apply_tts(text=text, sample_rate=self.sample_rate, speaker=self.speaker_name)
            return audio
        except Exception as e:
            raise ValueError(f"Synthesis error for text '{text}': {e}")

    def recognize_speech(self, audio_np: np.ndarray) -> RecognizeResult:
        """
        Recognize speech from the given audio data.

        Args:
            audio_np (np.ndarray): The audio data to recognize.

        Returns:
            RecognizeResult: The recognized segments with text.
        """
        try:
            language = lang_settings[self.language_from]['translation_key']
            result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language=language)
        except Exception as e:
            raise ValueError(f"Recognition error: {e}")

        return result

    def process_audio(self, timestamp: datetime, audio_data: bytes) -> Tuple[np.ndarray, Optional[Dict[str, str]]]:
        """
        Process the audio data by recognizing speech, translating text, and synthesizing speech.

        Args:
            timestamp (datetime): The timestamp of the audio data.
            audio_data (bytes): The audio data to process.

        Returns:
            Tuple[np.ndarray, Optional[Dict[str, str]]]: The final audio and log data.
        """
        audio_np, audio_length = self.normalize_audio(audio_data)

        recognized_result = self.recognize_speech(audio_np)
        recognized_segments = recognized_result['segments']
        recognized_language = recognized_result['language']
        synthesis_delay = (datetime.utcnow() - timestamp).total_seconds()

        if not recognized_segments or recognized_language == self.language_to:
            return audio_np, {
                "timestamp": timestamp,
                "original_text": recognized_result['text'],
                "translated_text": recognized_result['text'],
                "synthesis_delay": synthesis_delay,
                "recognize_result": recognized_result
            }

        translated_segments = []
        for segment in recognized_segments:
            no_speech_prob = segment['no_speech_prob']

            # Filter segments with low recognition probability
            if int(no_speech_prob) < 0.3:
                translated_text = self.translate_text(segment['text'])
            else:
                translated_text = ""

            translated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': translated_text
            })

        final_audio = np.array([])
        for translated_segment in translated_segments:
            translated_text = translated_segment['text']
            if translated_text:
                synthesized_segment = self.synthesize_speech(translated_text)
                silence_duration = int(translated_segment['start'] * self.sample_rate) - len(final_audio)
                if silence_duration > 0:
                    final_audio = np.pad(final_audio, (0, silence_duration), 'constant')
                final_audio = np.concatenate((final_audio, synthesized_segment), axis=None)

        synthesis_delay = (datetime.utcnow() - timestamp).total_seconds()

        original_text = " ".join([segment['text'] for segment in recognized_segments])
        translated_text = " ".join([segment['text'] for segment in translated_segments])

        log_data = {
            "timestamp": timestamp,
            "original_text": original_text,
            "translated_text": translated_text,
            "synthesis_delay": synthesis_delay,
            "recognize_result": recognized_result
        }

        return final_audio, log_data
