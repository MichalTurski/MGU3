import os
import librosa
import librosa.display
import numpy as np

def create_spectrogram(path):
    y, sr = librosa.load(path)
    return librosa.feature.melspectrogram(y=y, sr=sr)


def plot_spectrogram(spectrogram):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref = np.max), y_axis = 'mel', fmax = 8000, x_axis = 'time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()



def get_max_length():
    root_dir = './audio_dataset'
    out_dir = './spectrogram_dataset'
    max_len = 0

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for (top_dir, subdirectories, _) in os.walk(root_dir):
        for idx1, sub_dir in enumerate(subdirectories):
            print(f"I'm processing \"{sub_dir}\".")
            for (directory, _, files) in os.walk(os.path.join(top_dir, sub_dir)):
                for idx2, file in enumerate(files):
                    in_path = os.path.join(directory, file)
                    (filename, _) = os.path.splitext(file)

                    spectrogram = create_spectrogram(in_path)
                    curr_len = spectrogram.shape[1]
                    if curr_len > max_len:
                        max_len = curr_len
    print(f"max length = {max_len}")


def run():
    root_dir = './audio_dataset'
    out_dir = './spectrogram_dataset'
    description_file = 'All_files.csv'
    create_files_switch = True
    padding_switch = True
    max_len = 44 # It is max length of spectrogram

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    desc_file = open(os.path.join(out_dir, description_file), "w")
    for (top_dir, subdirectories, _) in os.walk(root_dir):
        for idx1, sub_dir in enumerate(subdirectories):
            print(f"I'm processing \"{sub_dir}\".")
            for (directory, _, files) in os.walk(os.path.join(top_dir, sub_dir)):
                for idx2, file in enumerate(files):
                    # if idx1 == idx2:
                    in_path = os.path.join(directory, file)
                    (filename, _) = os.path.splitext(file)
                    filename_extended = sub_dir + "_" + filename + ".npy"

                    if create_files_switch:
                        spectrogram = create_spectrogram(in_path)
                        if padding_switch:
                            padding = max_len - spectrogram.shape[1]
                            spectrogram = np.pad(spectrogram, ((0, 0), (padding, 0)), 'constant')

                        out_path = os.path.join(out_dir, filename_extended)

                        # plot_spectrogram(spectrogram)
                        # print(spectrogram.shape)

                        pickle_file = open(out_path, "wb")
                        np.save(pickle_file, spectrogram)
                        pickle_file.close()

                    desc_file.write(f"{filename_extended}, {sub_dir}\n")
    desc_file.close()


# get_max_length()
run()
