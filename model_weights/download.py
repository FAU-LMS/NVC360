import gdown


def download_one(file_id, target):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, target, quiet=False, fuzzy=True)


def main():
    file_ids = {
        '1fYXXJf9QsXg-teu44pkyB1BPJByZKz8k': 'checkpoint_dcvchem_vimeo90k.pth',
        '1eDEdguj7UU_e4VZ0UyHXEGMf_diETJxv': 'checkpoint_dcvchem_ugc360.pth',
        '1lsHVPawj5BRFgVhSDnNjMLWvcPyJ1gP0': 'checkpoint_dcvchem_ugc360+vimeo90k.pth',
        '1-CrfeZ9ovkFh4OyojpFKmKWc1dwCSBcr': 'checkpoint_dcvchem360_ugc360+vimeo90k.pth'
    }
    for file_id, target in file_ids.items():
        print("Downloading", target)
        download_one(file_id, target)
        print("Downloaded", target)


if __name__ == "__main__":
    main()
