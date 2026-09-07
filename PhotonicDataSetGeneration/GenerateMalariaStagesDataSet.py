import argparse
import os
import sys

import h5py
import torch

sys.path.append(os.getcwd())
from Utils.Utils import Config, get_device
from SensorStructure import MalariaPhotonicSensor


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Malaria Photonic Sensor Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def get_wavelengths(config, device):
    w_start, w_end = config.wavelength["range"]
    w_step = config.wavelength["step"]
    
    wavelengths = torch.arange(
        w_start,
        w_end + w_step,
        w_step,
        dtype=torch.float64,
        device=device
    )
    
    print(f"Wavelength range: {w_start} - {w_end} nm")
    print(f"Number of wavelengths: {len(wavelengths)}")
    return wavelengths


def distribute_samples(num_samples, stages):
    base_samples = num_samples // len(stages)
    remainder = num_samples % len(stages)
    
    samples_per_stage = [
        base_samples + 1 if i < remainder else base_samples 
        for i in range(len(stages))
    ]
    
    print("\nSamples per stage:")
    for stage, count in zip(stages, samples_per_stage):
        print(f"  {stage}: {count}")
        
    return samples_per_stage


def initialize_hdf5_metadata(f, config, wavelengths, num_samples):
    Nw = len(wavelengths)
    
    datasets = {
        "Reflectance": f.create_dataset("Reflectance", shape=(num_samples, Nw), dtype="float32"),
        "Transmittance": f.create_dataset("Transmittance", shape=(num_samples, Nw), dtype="float32"),
        "Absorption": f.create_dataset("Absorption", shape=(num_samples, Nw), dtype="float32"),
        "RI": f.create_dataset("RI", shape=(num_samples, Nw), dtype="float32"),
        "labels": f.create_dataset("labels", shape=(num_samples,), dtype="int8")
    }
    f.create_dataset("wavelengths", data=wavelengths.cpu().numpy())

    sensor_cfg = config.sensor
    f.attrs["sensor"] = "(Si/SiO2)^3/Ag/Defect/Ag/(Si/SiO2)^3"
    f.attrs["reference_wavelength_nm"] = sensor_cfg["reference_wavelength_nm"]
    f.attrs["angle_deg"] = sensor_cfg["angle_deg"]
    f.attrs["mode"] = sensor_cfg["mode"]
    f.attrs["metal"] = "Ag"
    f.attrs["metal_thickness_nm"] = sensor_cfg["metal_thickness_nm"]
    f.attrs["defect_thickness_nm"] = sensor_cfg["defect_thickness_nm"]
    f.attrs["si_periods"] = sensor_cfg["si_periods"]
    
    return datasets


def main():
    args = parse_arguments()
    config = Config.load(args.config)
    
    device = "cpu"
    print(f"Using device: {device}")

    wavelengths = get_wavelengths(config, device)

    save_path = config.data_set["save_generated_data_path"]
    stages = config.data_set["stages"]
    num_samples = config.data_set["num_samples"]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    samples_per_stage = distribute_samples(num_samples, stages)

    with h5py.File(save_path, "w") as f:
        datasets = initialize_hdf5_metadata(f, config, wavelengths, num_samples)
        
        current_idx = 0
        sensor_cfg = config.sensor

        for stage_idx, (stage, n_stage) in enumerate(zip(stages, samples_per_stage)):
            print(f"\nGenerating stage: {stage}")
            
            for sample_idx in range(n_stage):
                sensor = MalariaPhotonicSensor(
                    wavelengths_nm=wavelengths,
                    stage=stage,
                    angle_deg=sensor_cfg["angle_deg"],
                    mode=sensor_cfg["mode"],
                    si_periods=sensor_cfg["si_periods"],
                    reference_wavelength_nm=sensor_cfg["reference_wavelength_nm"],
                    metal_thickness_nm=sensor_cfg["metal_thickness_nm"],
                    defect_thickness_nm=sensor_cfg["defect_thickness_nm"],
                    device=device
                )
                
                result = sensor.simulate()
                
                datasets["Reflectance"][current_idx, :] = result["R"].detach().cpu().numpy()
                datasets["Transmittance"][current_idx, :] = result["T"].detach().cpu().numpy()
                datasets["Absorption"][current_idx, :] = result["A"].detach().cpu().numpy()
                datasets["RI"][current_idx, :] = result["RI"].detach().cpu().numpy()
                datasets["labels"][current_idx] = stage_idx
                
                current_idx += 1
                
                if (sample_idx + 1) % 100 == 0:
                    print(f"  {sample_idx + 1}/{n_stage}")
                    
            print(f"Finished {stage}")

    print("\n=================================")
    print("Dataset generation completed")
    print("=================================")
    print(f"Saved to: {save_path}")
    print(f"Total Samples: {num_samples}")
    print(f"Wavelengths: {len(wavelengths)}")


if __name__ == "__main__":
    main()