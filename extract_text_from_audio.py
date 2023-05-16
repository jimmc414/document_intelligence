import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the list of audio files in the directory
audio_dir = "c:\\python\\autoindex\\audio"
audio_files = os.listdir(audio_dir)

# Create a directory for the output text files
text_dir = "c:\\python\\autoindex\\txt_output"
os.makedirs(text_dir, exist_ok=True)

# Loop through the audio files and transcribe them
for audio_file in audio_files:
    # Open the audio file as a binary file
    audio_path = os.path.join(audio_dir, audio_file)
    with open(audio_path, "rb") as f:
        # Call the OpenAI Audio endpoint to transcribe the file
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=f,
            language="en",
            response_format="verbose_json"
        )
    
    # Get the text from the transcript
    text = transcript["text"]

    # Write the text to a file with the same name as the audio file
    text_file = audio_file.split(".")[0] + ".txt"
    text_path = os.path.join(text_dir, text_file)
    with open(text_path, "w") as f:
        f.write(text)