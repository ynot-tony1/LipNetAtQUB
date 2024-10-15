import os
from google.cloud import speech_v1p1beta1 as speech

def transcribe_audio_with_timestamps(audio_file_path):
    # Initialize the Speech-to-Text client
    client = speech.SpeechClient()

    # Read the audio file
    with open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()

    # Configure the audio settings
    audio = speech.RecognitionAudio(content=content)

    # Configure recognition settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # Ensure this matches the sample rate of your audio file
        language_code='en-US',
        enable_word_time_offsets=True,
        model='default',  # You can specify enhanced models if needed
    )

    # For longer audio files, use long_running_recognize
    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=300)

    transcription = ""
    words_info = []

    # Extract the transcribed text and word-level timestamps
    for result in response.results:
        alternative = result.alternatives[0]
        transcription += alternative.transcript + ' '

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()
            words_info.append({
                'word': word,
                'start_time': start_time,
                'end_time': end_time
            })

    return transcription.strip(), words_info

if __name__ == "__main__":
    # Path to the extracted audio file
    audio_file = 'C:/Users/tony_/lipnet training videos mpg/output_audio.wav'
    transcription, words_info = transcribe_audio_with_timestamps(audio_file)

# Output the transcribed text and word-level timestamps
print("Transcription:")
print(transcription)
print("\nWord-level timestamps:")
for word_info in words_info:
    print(f"Word: {word_info['word']}, "
          f"Start Time: {word_info['start_time']:.3f}s, "
          f"End Time: {word_info['end_time']:.3f}s")