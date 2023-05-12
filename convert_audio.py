import os
from pydub import AudioSegment

def convert_amr_to_wav(amr_file, wav_file):
    audio = AudioSegment.from_file(amr_file, format='amr')
    audio.export(wav_file, format='wav')

def main():
    source_path = 'C:/python/autoindex/audio/'
    target_path = 'C:/python/autoindex/audio/'

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for filename in os.listdir(source_path):
        if filename.endswith('.amr'):
            amr_file = os.path.join(source_path, filename)
            basename = os.path.splitext(filename)[0]
            wav_file = os.path.join(target_path, basename + '.wav')
            convert_amr_to_wav(amr_file, wav_file)
            print(f'Converted: {filename} to {basename}.wav')

if __name__ == '__main__':
    main()