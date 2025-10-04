from pathlib import Path
from .utils import MODEL_NAME_TO_ID, PROJECTION_NAME_TO_ID, WEIGHTS_BASEFOLDER, write_header, hash_weights, get_png_info
from .DCVCHEMController import DCVCHEMController
from .DCVCHEM360Controller import DCVCHEM360Controller


def encode(args):
    print("Encoder configuration:")
    print(f"  Model:         {args.model}")
    print(f"  Intra weights: {args.intra_weights}")
    print(f"  Inter weights: {args.inter_weights}")
    print(f"  Quality:       {args.quality}")
    print(f"  Projection:    {args.projection}")
    print(f"  GOP:           {args.gop}")
    print(f"  Input:         {args.input}")
    print(f"  Output:        {args.output}")

    if not Path(args.input).exists() or not Path(args.input).is_dir():
        raise FileNotFoundError(f"Input PNG folder does not exist: {args.input}")
    frames, width, height, bitdepth = get_png_info(args.input)
    print(f"  Frames:        {frames}")
    print(f"  Width:         {width}")
    print(f"  Height:        {height}")
    print(f"  Bitdepth:      {bitdepth}")


    if args.model not in MODEL_NAME_TO_ID:
        raise ValueError(f"Invalid model '{args.model}'")
    model_id = MODEL_NAME_TO_ID[args.model]

    quality = args.quality
    if quality < 0 or quality > 4:
        raise ValueError("Quality index out of range.")
    
    if args.projection not in PROJECTION_NAME_TO_ID:
        raise ValueError(f"Invalid projection '{args.projection}'")
    projection_id = PROJECTION_NAME_TO_ID[args.projection]

    gop = args.gop
    if gop < 1:
        raise ValueError("GOP must be a positive value.")

    intra_weights = WEIGHTS_BASEFOLDER / args.intra_weights
    if not intra_weights.exists() or not intra_weights.is_file():
        raise FileNotFoundError(f"Intra weights file does not exist: {intra_weights}")

    inter_weights = WEIGHTS_BASEFOLDER / args.inter_weights
    if not inter_weights.exists() or not inter_weights.is_file():
        raise FileNotFoundError(f"Inter weights file does not exist: {inter_weights}")

    with open(args.output, "wb") as f:
        write_header(f, frames, width, height, bitdepth, projection_id, quality, gop, model_id, hash_weights(intra_weights), hash_weights(inter_weights))

        if args.model == 'dcvchem':
            controller = DCVCHEMController(intra_weights, inter_weights, quality, gop, args.device)
        elif args.model == 'dcvchem360':
            controller = DCVCHEM360Controller(intra_weights, inter_weights, quality, gop, args.projection, args.device)
        else:
            raise ValueError(f"Invalid model '{args.model}'")
        
        controller.encode(args.input, f, width, height, bitdepth, frames)
