from pathlib import Path
from .utils import find_weights, read_header, MODEL_IDS, PROJECTION_IDS
from .DCVCHEMController import DCVCHEMController
from .DCVCHEM360Controller import DCVCHEM360Controller


def decode(args):
    with open(args.input, "rb") as f:
        header, header_size = read_header(f)

        if args.header_only:
            print(f"Header info ({header_size} bytes):")
            maxlen = max(len(k) for k in header.keys())
            for key, value in header.items():
                print(f"  {key:<{maxlen}}: {value}")
            return

        model_name = MODEL_IDS.get(header["model_id"], f"unknown({header['model_id']})")
        projection_name = PROJECTION_IDS.get(header["projection_id"], f"unknown({header['projection_id']})")
        intra_weights = find_weights(header['intra_weights_hash'], label='intra')
        inter_weights = find_weights(header['inter_weights_hash'], label='inter')

        print("Decoder configuration:")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print("Header info:")
        print(f"  Frames:        {header['frames']}")
        print(f"  Width:         {header['width']}")
        print(f"  Height:        {header['height']}")
        print(f"  Bitdepth:      {header['bitdepth']}")
        print(f"  Quality:       {header['quality']}")
        print(f"  Projection:    {projection_name}")
        print(f"  GOP:           {header['gop']}")
        print(f"  Model:         {model_name}")
        print(f"  Intra weights: {intra_weights}")
        print(f"  Inter weights: {inter_weights}")

        if model_name == 'dcvchem':
            controller = DCVCHEMController(intra_weights, inter_weights, header['quality'], header['gop'], args.device)
        elif model_name == 'dcvchem360':
            controller = DCVCHEM360Controller(intra_weights, inter_weights, header['quality'], header['gop'], projection_name, args.device)
        else:
            raise ValueError(f"Invalid model '{model_name}'")
        
        controller.decode(f, args.output, header['width'], header['height'], header['bitdepth'], header['frames'])
