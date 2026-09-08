# Two-Sensor Structure and Results Report

This document summarizes the two malaria sensor configurations used in the benchmark notebooks and the resonance analysis results extracted from the generated datasets.

The analysis notebooks are:

- [analysis_dataset_sensor_1.ipynb](analysis_dataset_sensor_1.ipynb)
- [analysis_dataset_sensor_2.ipynb](analysis_dataset_sensor_2.ipynb)

The corresponding CSV outputs are stored in [Results](Results/).

## Sensor 1 Structure

Sensor 1 is configured for the visible/near-infrared range and uses a single-resonance photonic structure.

| Parameter | Value |
| --- | --- |
| Wavelength range | 700 to 900 nm |
| Wavelength step | 0.1 nm |
| Reference wavelength | 800.0 nm |
| Si periods | 3 |
| Metal thickness | 15.0 nm |
| Defect thickness | 7000.0 nm |
| Incident angle | 86.0 degrees |
| Polarization mode | TM |
| Dataset path | `data/malaria_1dpc_2022.h5` |
| Number of samples | 4000 |
| Stages | normal, ring, trophozoite, schizont |

## Sensor 2 Structure

Sensor 2 is configured for the infrared range and uses a coupled-resonator structure.

| Parameter | Value |
| --- | --- |
| Wavelength range | 1600 to 2200 nm |
| Wavelength step | 0.1 nm |
| Coupled resonators | N = 3, M = 2 |
| Si thickness | 163.0 nm |
| SiO2 thickness | 250.0 nm |
| Defect thickness | 2516.0 nm |
| Incident angle | 85.0 degrees |
| Polarization mode | TM |
| Reference stage | normal |
| Dataset path | `Data/malaria_sensor2.h5` |
| Number of samples | 4000 |
| Stages | normal, ring, trophozoite, schizont |

## Result Summary

The benchmark analysis tracked the resonance closest to a target wavelength for each stage and then computed the sensitivity from the resonance shift and refractive-index change relative to the normal stage.

### Sensor 1 Results

| Stage | Resonance wavelength (nm) | Transmission | RI |
| --- | ---: | ---: | ---: |
| Normal | 793.1 | 0.70839065 | 1.3707981 |
| Ring | 789.8 | 0.71034400 | 1.3678435 |
| Trophozoite | 784.1 | 0.71291220 | 1.3627605 |
| Schizont | 779.3 | 0.71433794 | 1.3585074 |

| Stage | $\Delta\lambda$ (nm) | $\Delta RI$ | Sensitivity (nm/RIU) |
| --- | ---: | ---: | ---: |
| Ring | -3.3 | -0.0029546022 | 1116.90 |
| Trophozoite | -9.0 | -0.0080375670 | 1119.74 |
| Schizont | -13.8 | -0.0122907160 | 1122.80 |

### Sensor 2 Results

| Stage | Resonance wavelength (nm) | Transmission | RI |
| --- | ---: | ---: | ---: |
| Normal | 1997.6 | 1.00000000 | 1.3648093 |
| Ring | 1994.9 | 0.99999610 | 1.3618050 |
| Trophozoite | 1990.1 | 0.99992650 | 1.3566343 |
| Schizont | 1986.2 | 0.99981683 | 1.3523070 |

| Stage | $\Delta\lambda$ (nm) | $\Delta RI$ | Sensitivity (nm/RIU) |
| --- | ---: | ---: | ---: |
| Ring | -2.7 | -0.0030043125 | 898.71 |
| Trophozoite | -7.5 | -0.0081750150 | 917.43 |
| Schizont | -11.4 | -0.0125023130 | 911.83 |

## Observations

- Both sensors show a monotonic blue shift in resonance wavelength as the malaria stage progresses from normal to schizont.
- Sensor 1 exhibits a stronger wavelength shift per refractive-index unit than Sensor 2, with sensitivity values around 1117 to 1123 nm/RIU compared with about 899 to 917 nm/RIU for Sensor 2.
- Sensor 2 operates at higher wavelengths and its normal-state transmission is essentially unity, while Sensor 1 has a lower peak transmission but stronger stage-dependent wavelength displacement.

## Output Files

- [Results/resonance_analysis_sensor_1.csv](Results/resonance_analysis_sensor_1.csv)
- [Results/sensitivity_analysis_sensor_1.csv](Results/sensitivity_analysis_sensor_1.csv)
- [Results/resonance_analysis_sensor_2.csv](Results/resonance_analysis_sensor_2.csv)
- [Results/sensitivity_analysis_sensor_2.csv](Results/sensitivity_analysis_sensor_2.csv)
