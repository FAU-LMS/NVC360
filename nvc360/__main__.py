import argparse
from pathlib import Path
import torch
from .encode import encode
from .decode import decode


def main():
    parser = argparse.ArgumentParser(
        description="Neural 360-Degree Video Compression (NVC360)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument(
            "--device",
            default="cuda:0" if torch.cuda.is_available() else "cpu",
            help="Device to use for encoding/decoding"
        )

    encode_parser = subparsers.add_parser("encode", help="Encode a sequence")
    add_common(encode_parser)
    encode_parser.add_argument(
        "--model",
        choices=["dcvchem", "dcvchem360"],
        required=True,
        help="Model architecture to use."
    )
    encode_parser.add_argument(
        "--intra-weights",
        required=True,
        help="Name of intra model weights file."
    )
    encode_parser.add_argument(
        "--inter-weights",
        type=Path,
        required=True,
        help="Name of inter model weights file."
    )
    encode_parser.add_argument(
        "--quality",
        type=int,
        choices=[0, 1, 2, 3],
        required=True,
        help="Quality parameter."
    )
    encode_parser.add_argument(
        "--projection",
        choices=["erp", "none"],
        required=True,
        help="Projection format of input video or none if classical perspective video."
    )
    encode_parser.add_argument(
        "--gop",
        type=int,
        default=32,
        help="Group-of-pictures size (intra refresh period)."
    )
    encode_parser.add_argument(
        "input",
        type=Path,
        help="Input PNG folder."
    )
    encode_parser.add_argument(
        "output",
        type=Path,
        help="Output bitstream file."
    )
    encode_parser.set_defaults(func=encode)

    decode_parser = subparsers.add_parser("decode", help="Decode a sequence")
    add_common(decode_parser)
    decode_parser.add_argument(
        "--header-only",
        action='store_true',
        help="Print the info in the bitstream header and exit."
    )
    decode_parser.add_argument(
        "input",
        type=Path,
        help="Input bitstream file."
    )
    decode_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output PNG folder."
    )
    decode_parser.set_defaults(func=decode)

    args = parser.parse_args()

    if args.command == "decode" and not args.header_only and args.output is None:
        decode_parser.error("the following argument is required for decode (unless --header-only): output")

    args.func(args)


if __name__ == "__main__":
    main()
