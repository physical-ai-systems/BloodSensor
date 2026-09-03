import argparse
import sys
import os
sys.path.append(os.getcwd())
from click import parser
import h5py as h5  
from Utils.Utils import Config, get_device
import torch
import yaml
from sensor_structure import PhotonicSensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    config = Config.load(args.config)
    device = get_device()
    print(f"Using device: {device}")
    wavelengths = config.wavelength["range"]
    save_path = config.data_set["save_generated_data_path"]
    stages = config.data_set["stages"]
    num_samples = config.data_set["num_samples"]
    sample_per_stage = num_samples // len(stages)
    batch_size = config.data_set["batch_size"]
    perturbation = config.structure["perturbation"]

    wavelength = torch.arange(wavelengths[0], wavelengths[1],config.wavelength["step"])    
    with h5.File(save_path, "w") as f:
        dset_T = f.create_dataset("Transmittance", shape=(num_samples, len(wavelength)), dtype='float32')
        dset_R = f.create_dataset("Reflectance", shape=(num_samples, len(wavelength)), dtype='float32')
        dset_L = f.create_dataset("labels", shape=(num_samples,), dtype='int8')
        f.create_dataset("wavelengths", data=wavelength.cpu().numpy())
        current_idx = 0
        for stage_idx, stage in enumerate(stages):
            for i in range(sample_per_stage):
                sensor = PhotonicSensor(N = config.structure["N"], angle=config.structure["angle"],mode=config.structure["mode"], device=device, materials=config.materials, thicknesses=config.thicknesses, wavelength_nm=wavelength, stage = stage)
                sensor.perturbate_structure_thickness(perturbation)
                T, R = sensor.TransferMatrixMethod(stage = stage)
                dset_T[current_idx, :] = T.cpu().numpy()
                dset_R[current_idx, :] = R.cpu().numpy()
                dset_L[current_idx] = stage_idx
                current_idx += 1
                if (i + 1) % batch_size == 0:
                    print(f"Generated {i + 1} samples for stage '{stage}'")
            print(f"Finished generating samples for stage '{stage}'")


if __name__ == "__main__":
    main()

