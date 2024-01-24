from pathlib import Path

# =========================================================
# File Path
# =========================================================
Dataset = {
    "Path": (Path(__file__).parents[2] / "data"),
    "FileName": "processed_data.csv",
    "train_test_split_point": 0.0,
}

Paths = {
    "Results": (Path(__file__).parents[2] / "results"),
    "Plots": (Path(__file__).parents[2] / "plots"),
}

# =========================================================
# Global Parameters
# =========================================================
PhysicalProperties = {
    "Density": 963,
    "HeatCapacity": 4181.3,
    "MaxTemp": 180,
    "MinSupTemp": 100,
    "EnergyUnitConversion": 10 ** 6,  # The scale to convert given energy unit to Watt
}

GridProperties = {"ConsumerNum": 1}

GridProperties["PipeNum"] = (
    2 * GridProperties["ConsumerNum"]
    if GridProperties["ConsumerNum"] == 1
    else 2 * (GridProperties["ConsumerNum"] + 1)
)

TimeParameters = {
    "TimeInterval": 3600,  # seconds
    "PlanningHorizon": 24,  # steps
    "ActionHorizon": 1,  # steps
}

# =========================================================
# Grid Parameters
# =========================================================
# generator scale muliplier
gsm = 1

Generator1 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.12*gsm,0],[0.75*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.81,
    "FuelCost": [1, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":505.1*gsm,
}

Generator2 = Generator1
Generator3 = Generator1

Generator4 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.12*gsm,0],[0.47*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.89,
    "FuelCost": [1.64, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":154*1.64*gsm,
}

Generator5 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.79*gsm,0],[2.74*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.89,
    "FuelCost": [5.86, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":43.1*5.86*gsm,
}

Generator6 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.39*gsm,0],[1.73*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.89,
    "FuelCost": [6.34, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":4*6.34*gsm,
}

Generator7 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.55*gsm,0],[1.2*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.89,
    "FuelCost": [6.34, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":2*6.34*gsm,
}

Generator8 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.2*gsm,0],[1.14*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.87,
    "FuelCost": [8.33, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":3*8.33*gsm,
}

Generator9 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.2*gsm,0],[1.37*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.87,
    "FuelCost": [8.51, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":3*8.51*gsm,
}

Generator10 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.27*gsm,0],[2.04*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.87,
    "FuelCost": [8.59, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":2.9*8.59*gsm,
}

Generator11 = Generator10

Generator12 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.12*gsm,0],[1.14*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.87,
    "FuelCost": [9.6, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":1.3*9.6*gsm,
}

Generator13 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.12*gsm,0],[1.37*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.87,
    "FuelCost": [9.82, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":1.3*9.82*gsm,
}

Generator14 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.0*gsm,0],[0.98*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.87,
    "FuelCost": [14.17, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":0.89*14.17*gsm,
}

Generator15 = {
    "CHPType": "keypts",
    "OperationRegion": [[0.0*gsm,0],[1.37*gsm,0]],  # Heat, power. MW
    "Efficiency": 0.87,
    "FuelCost": [14.17, 0],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": -1,
    "StartupCost":0.89*14.17*gsm,
}
CHPPreset1 = {
    "CHPType": "keypts",
    "OperationRegion": [[10, 5], [0, 10], [0, 50], [70, 35]],  # Heat, power. MW
    "Efficiency": 1,
    "FuelCost": [8.1817, 38.1805],
    "MaxRampRateQ": -1,
    "MaxRampRateE": -1,
    "MaxRampRateTemp": 3,
}

ProducerPreset1 = {
    "Type": "CHP",
    "Parameters": CHPPreset1,
    "PumpEfficiency": 1,
    "ControlWithTemp": True,
}

ConsumerPreset1 = {
    "k": 5*10**6/400*(400**(-0.8)+400**(-0.8)),
    "q": 0.8,
    "SurfaceArea": 135,
    "TempReturnSeconary": 60,
    "SetPointTempSupplySecondary": 90,
    "MaxMassFlowPrimary": 805.15,
    "MinTempSupplyPrimary": 100,
    "FixPressureLoad": 100000,
}

PipePreset1 = {
    "Diameter": 0.25,
    "Length": 9000,
    "ThermalResistance": 1.97*0.6,  # in k*m/W, = 1/U
    "InitialTemperature": 130,
    "EnvironmentTemperature": 10,
    "MaxFlowSpeed": 3,
    "MinFlowSpeed": 0,
    "FrictionCoefficient": 1.29 * 1.414,
}

PipePreset2 = {
    "Diameter": 0.25,
    "Length": 9000,
    "ThermalResistance": 1.97*0.6,  # in k*m/W, = 1/U
    "InitialTemperature": 80,
    "EnvironmentTemperature": 10,
    "MaxFlowSpeed": 3,
    "MinFlowSpeed": 0,
    "FrictionCoefficient": 1.29 * 1.414,
}

# supply side edge
PipePreset3 = {
    "Diameter": 0.2,
    "Length": 3000,
    "ThermalResistance": 1.1,  # in k*m/W, = 1/U
    "InitialTemperature": 110,
    "EnvironmentTemperature": 10,
    "MaxFlowSpeed": 3,
    "MinFlowSpeed": 0,
    "FrictionCoefficient": 1.29 * 2,
}

# return side edge
PipePreset4 = {
    "Diameter": 0.2,
    "Length": 3000,
    "ThermalResistance": 1.1,  # in k*m/W, = 1/U
    "InitialTemperature": 50,
    "EnvironmentTemperature": 10,
    "MaxFlowSpeed": 3,
    "MinFlowSpeed": 0,
    "FrictionCoefficient": 1.29 * 2,
}
# GridParameters = {
#   'Producers': [
#     {
#       'Type': 'CHP',
#       'Parameters': CHPPreset1,
#       'PumpEfficiency': 1,
#       'ControlWithTemp': True,
#     }
#   ],
#   'Consumers': [
#     HXPreset1,
#   ],
#   'Pipes': [
#     PipePreset1,
#     PipePreset2,
#   ],
# }
