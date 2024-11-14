import argparse
import os

import torch

from uni2ts.model.moirai import MoiraiModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        choices=[
            "moirai-1.0-R-small",
            "moirai-1.0-R-base",
            "moirai-1.0-R-large",
            "moirai-1.1-R-small",
            "moirai-1.1-R-base",
            "moirai-1.1-R-large",
        ],
        help="Choose the pretrained Moirai model version",
    )

    args = parser.parse_args()

    # Load the pretrained Moirai model
    pretrained_moirai = MoiraiModule.from_pretrained("Salesforce/" + args.model_name)

    # Define the directory and checkpoint path
    save_dir = "./ckpt/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the model state_dict as a .ckpt file in the specified directory
    model_name = args.model_name.replace("-", "_")
    checkpoint_path = os.path.join(save_dir, model_name + ".ckpt")
    torch.save(pretrained_moirai.state_dict(), checkpoint_path)

    print(f"Model saved to {checkpoint_path}")
