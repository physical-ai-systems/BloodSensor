import yaml
import torch
import h5py
import argparse
import os
from FirstStage import FirstStage

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def generate_transmittance_curve(config, ngtype, ndtype):
    layers = config['layers']
    distances = config['distances']
    refractive_indices = config['refractive_indices']
    wavelen_cfg = config['wavelen']

    wavelen_start, wavelen_end = wavelen_cfg['wavelen_range']
    n_steps = int(round((float(wavelen_end) - float(wavelen_start)) / 0.5)) + 1
    wavelengths = torch.linspace(
        float(wavelen_start),
        float(wavelen_end),
        n_steps,
        dtype=torch.float64,
    )

    transmittance_values = []
    with torch.no_grad():
        for wavelength in wavelengths:
            stage = FirstStage(
                layar_a=layers['layar_a'],
                layar_b=layers['layar_b'],
                layar_g=layers['layar_g'],
                layar_d=ndtype,
                d1=float(distances['d1']),
                d2=float(distances['d2']),
                dg=float(distances['dg']),
                dd=float(distances['dd']),
                n1=float(refractive_indices['n1']),
                n2=float(refractive_indices['n2']),
                wavelen=wavelength,
                suround_in=layers['suround_in'],
                suround_out=layers['suround_out'],
                N=int(layers['N']),
                nAir=float(refractive_indices['nAir']),
                nSiO2=float(refractive_indices['nSiO2']),
                device=config['device'],
            )
            transmittance_values.append(float(stage.run(ngtype=ngtype, ndtype=ndtype).item()))

    return wavelengths, transmittance_values


def build_conditions(config):
    layers = config['layers']
    ngtypes = layers['ngtype']
    ndtypes = layers['ndtype']

    conditions = []
    for ndtype in ndtypes:
        for ngtype in ngtypes:
            conditions.append((ndtype, ngtype))
    return conditions


def save_dataset(output_path, wavelengths, conditions, transmittance_matrix):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('wavelength_nm', data=wavelengths)
        h5f.create_dataset('transmittance', data=transmittance_matrix)

        ndtype_values = [c[0] for c in conditions]
        ngtype_values = [c[1] for c in conditions]
        h5f.create_dataset('ndtype', data=ndtype_values)
        h5f.create_dataset('ngtype', data=ngtype_values)

        curves_group = h5f.create_group('curves')
        for index, (ndtype, ngtype) in enumerate(conditions):
            ds_name = f"{ndtype}_{ngtype}"
            curves_group.create_dataset(ds_name, data=transmittance_matrix[index])

        h5f.attrs['schema'] = 'first_stage_transmittance_v1'
        h5f.attrs['samples'] = int(wavelengths.shape[0])
        h5f.attrs['num_conditions'] = int(len(conditions))

def main():
    parser = argparse.ArgumentParser(description="Run First Stage")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)

    output_path = config['dataset']['path']
    conditions = build_conditions(config)
    sample = int(config['dataset']['sample'])
    for _ in range(sample):
        print(f"Running sample {_+1}/{sample}")
        wavelengths = None
        curves = []
        for ndtype, ngtype in conditions:
            wl, t_curve = generate_transmittance_curve(config, ngtype=ngtype, ndtype=ndtype)
            if wavelengths is None:
                wavelengths = wl
            curves.append(t_curve)
        transmittance_matrix = torch.vstack([torch.tensor(c, dtype=torch.float64) for c in curves])
        save_dataset(output_path, wavelengths, conditions, transmittance_matrix)

    print(f"Saved dataset to: {output_path}")
    print(f"Wavelength samples: {wavelengths.shape[0]}")
    print(f"Generated conditions: {len(conditions)}")
    for ndtype, ngtype in conditions:
        print(f" - {ndtype} ({ngtype})")

if __name__ == "__main__":
    main()