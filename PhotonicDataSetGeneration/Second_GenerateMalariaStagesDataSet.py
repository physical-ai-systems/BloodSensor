import argparse
import os
import sys
import h5py
import torch
sys.path.append(os.getcwd())
from Utils.Utils import Config
from SensorStructure import CoupledResonatorMalariaSensor
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Sensor 2 Malaria Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to Sensor 2 configuration file")
    return parser.parse_args()
def get_wavelengths(config, device):
    w_start, w_end = config.wavelength["range"]
    w_step = config.wavelength["step"]
    return torch.arange(w_start, w_end + w_step, w_step, dtype=torch.float64, device=device)
def distribute_samples(num_samples, stages):
    base_samples = num_samples // len(stages)
    remainder = num_samples % len(stages)
    return [base_samples + 1 if i < remainder else base_samples for i in range(len(stages))]
def initialize_hdf5(f, config, wavelengths, num_samples):
    Nw = len(wavelengths)
    sensor_cfg = config.sensor
    datasets = {
        "Reflectance": f.create_dataset("Reflectance", shape=(num_samples, Nw), dtype="float32"),
        "Transmittance": f.create_dataset("Transmittance", shape=(num_samples, Nw), dtype="float32"),
        "Absorption": f.create_dataset("Absorption", shape=(num_samples, Nw), dtype="float32"),
        "RI": f.create_dataset("RI", shape=(num_samples, Nw), dtype="float32"),
        "labels": f.create_dataset("labels", shape=(num_samples,), dtype="int8")
    }
    datasets["RI_reference"] = f.create_dataset("RI_reference", shape=(num_samples, Nw), dtype="float32")
    f.create_dataset("wavelengths", data=wavelengths.cpu().numpy())
    f.attrs["sensor"] = "Air/SiO2/(Si/SiO2)^N/D1/(SiO2/Si)^M/SiO2/D2/SiO2/(Si/SiO2)^N/Air"
    f.attrs["sensor_type"] = "CoupledResonatorMalariaSensor"
    f.attrs["N"] = sensor_cfg["N"]
    f.attrs["M"] = sensor_cfg["M"]
    f.attrs["angle_deg"] = sensor_cfg["angle_deg"]
    f.attrs["mode"] = sensor_cfg["mode"]
    f.attrs["si_thickness_nm"] = sensor_cfg["si_thickness_nm"]
    f.attrs["sio2_thickness_nm"] = sensor_cfg["sio2_thickness_nm"]
    f.attrs["defect_thickness_nm"] = sensor_cfg["defect_thickness_nm"]
    f.attrs["reference_stage"] = sensor_cfg.get("reference_stage", "normal")
    return datasets
def create_sensor(wavelengths, stage, sensor_cfg, device):
    return CoupledResonatorMalariaSensor(wavelengths_nm=wavelengths, stage=stage, angle_deg=sensor_cfg["angle_deg"], mode=sensor_cfg["mode"], N=sensor_cfg["N"], M=sensor_cfg["M"], si_thickness_nm=sensor_cfg["si_thickness_nm"], sio2_thickness_nm=sensor_cfg["sio2_thickness_nm"], defect_thickness_nm=sensor_cfg["defect_thickness_nm"], reference_stage=sensor_cfg.get("reference_stage", "normal"), device=device)
def main():
    args = parse_arguments()
    config = Config.load(args.config)
    sensor_cfg = config.sensor
    device = "cpu"
    wavelengths = get_wavelengths(config, device)
    save_path = config.data_set["save_generated_data_path"]
    stages = config.data_set["stages"]
    num_samples = config.data_set["num_samples"]
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    samples_per_stage = distribute_samples(num_samples, stages)
    with h5py.File(save_path, "w") as f:
        datasets = initialize_hdf5(f, config, wavelengths, num_samples)
        current_idx = 0
        for stage_idx, (stage, n_stage) in enumerate(zip(stages, samples_per_stage)):
            for sample_idx in range(n_stage):
                sensor = create_sensor(wavelengths=wavelengths, stage=stage, sensor_cfg=sensor_cfg, device=device)
                result = sensor.simulate()
                datasets["Reflectance"][current_idx, :] = result["R"].detach().cpu().numpy()
                datasets["Transmittance"][current_idx, :] = result["T"].detach().cpu().numpy()
                datasets["Absorption"][current_idx, :] = result["A"].detach().cpu().numpy()
                datasets["RI"][current_idx, :] = result["RI"].detach().cpu().numpy()
                datasets["RI_reference"][current_idx, :] = result["reference_RI"].detach().cpu().numpy()
                datasets["labels"][current_idx] = stage_idx
                current_idx += 1
                if (sample_idx + 1) % 100 == 0:
                    print(f"  {sample_idx + 1}/{n_stage}")
                    
            print(f"Finished {stage}")
if __name__ == "__main__":
    main()