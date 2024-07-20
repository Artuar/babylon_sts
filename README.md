# Babylon STS

A library for audio processing with speech recognition and translation.

## Installation

```bash
pip install git+https://github.com/Artuar/babylon_sts.git
```

## Usage examples

### Processing a Local Audio File

Here is an example of how to process a local audio file, translate its content, and save the result to a new file:

```
python
Copy code
import os
import numpy as np
import soundfile as sf
from datetime import datetime
from audio_processor.processor import AudioProcessor

def process_local_audio(input_file: str, output_file: str, language: str = 'ua', model_name: str = 'medium', sample_rate: int = 24000):
    # Create an instance of AudioProcessor with the required parameters
    audio_processor = AudioProcessor(language=language, model_name=model_name, sample_rate=sample_rate)

    ## Read the audio file
    audio_data, file_sample_rate = sf.read(input_file, dtype='int16')
    audio_data = audio_data.T.tobytes()  # Convert data to bytes
    
    # Current time as timestamp for processing
    timestamp = datetime.utcnow()
    
    # Process the audio data
    final_audio, log_data = audio_processor.process_audio(timestamp, audio_data)
    
    # Save the processed audio to a new file
    sf.write(output_file, final_audio, sample_rate)

# Call the function to process a local file
process_local_audio('input_audio.wav', 'translated_audio.wav')

```

### Args:
- language (str): The language code. Possible values: 'en', 'ua', 'ru', 'fr', 'de', 'es', 'hi'.
- model_name (str): The Whisper model to use. Possible values: 'tiny', 'base', 'small', 'medium', 'large'.
- sample_rate (int): The sample rate for audio processing.
- speaker (Optional[str]): The name of speaker for speech synthesize. Full speakers list here https://github.com/snakers4/silero-models?tab=readme-ov-file#models-and-speakers


## Install requirements

```bash
pip install -r requirements.txt
```

## Tests

```bash
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.