import os
from moviepy.editor import AudioFileClip
from openai import OpenAI
import math

client = OpenAI()
# Function to split audio file
def split_audio(file_path, chunk_length_seconds=600):  # 10 minutes chunks
    audio = AudioFileClip(file_path)
    duration = audio.duration
    chunks = math.ceil(duration / chunk_length_seconds)

    for i in range(chunks):
        start_time = i * chunk_length_seconds
        end_time = min((i+1) * chunk_length_seconds, duration)
        chunk = audio.subclip(start_time, end_time)
        chunk.write_audiofile(f"chunk_{i}.mp3")

    audio.close()
    return chunks

# Function to transcribe audio chunk
def transcribe_chunk(file_path, prompt=''):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
            response_format="text",
            prompt=prompt
        )
    return transcription

# Function to get the last n words of a transcript
def get_last_n_words(text, n=50):
    return ' '.join(text.split()[-n:])

# Main process
def main():
    input_file = r"V:\Prague_Biking\Data\Aart Interview CROW.m4a.mp3"
    output_filename = "Aart_CROW_Interview.txt"

    # Split the audio file
    num_chunks = split_audio(input_file)

    # Transcribe each chunk and combine
    full_transcription = ""
    prev_chunk_end = ""  # Store the end of the previous chunk's transcription

    for i in range(num_chunks):
        chunk_file = f"chunk_{i}.mp3"
        chunk_transcription = transcribe_chunk(chunk_file, prompt=prev_chunk_end)
        full_transcription += chunk_transcription + "\n\n"

        # Get the last few words to use as prompt for next chunk
        prev_chunk_end = get_last_n_words(chunk_transcription, n=50)

        os.remove(chunk_file)  # Clean up the chunk file

    # Save the full transcription
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_transcription)

    print(f"Transcription complete. Output saved to {output_filename}.")

if __name__ == "__main__":
    main()
