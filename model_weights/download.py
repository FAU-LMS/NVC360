from pathlib import Path
import gdown
import urllib.request


def download_intra(target):
    print("Downloading", target)
    print("\n### !!! WARNING !!! ###\nAutomatic download of DCVC-HEM intra model weights is currently broken due to authentication issues.\nIn the meantime, please download the acmmm2022_image_psnr.pth.tar weights from this page manually and put them into the model_weights folder in this repository:\nhttps://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBb3pmVlZ3dFdXWW9pVUFHazZ4ci1vRUxib2RuP2U9a3J5Mk5r&id=2866592D5C55DF8C%211216&cid=2866592D5C55DF8C\n### !!! WARNING !!! ###\n")
    # urllib.request.urlretrieve(
    #     'https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211220&authkey=AMRg1W3PVt_F3yc',
    #     target
    # )
    # print("Downloaded", target)


def download_one(file_id, target):
    print("Downloading", target)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, target, quiet=False, fuzzy=True)
    # print("Downloaded", target)


def main():
    target = str(Path(__file__).parent / 'acmmm2022_image_psnr.pth.tar')
    download_intra(target)

    file_ids = {
        '1fYXXJf9QsXg-teu44pkyB1BPJByZKz8k': 'checkpoint_dcvchem_vimeo90k.pth',
        '1eDEdguj7UU_e4VZ0UyHXEGMf_diETJxv': 'checkpoint_dcvchem_ugc360.pth',
        '1lsHVPawj5BRFgVhSDnNjMLWvcPyJ1gP0': 'checkpoint_dcvchem_ugc360+vimeo90k.pth',
        '1-CrfeZ9ovkFh4OyojpFKmKWc1dwCSBcr': 'checkpoint_dcvchem360_ugc360+vimeo90k.pth'
    }
    for file_id, target in file_ids.items():
        target = Path(__file__).parent / target
        if target.exists():
            continue
        download_one(file_id, str(target))


if __name__ == "__main__":
    main()
