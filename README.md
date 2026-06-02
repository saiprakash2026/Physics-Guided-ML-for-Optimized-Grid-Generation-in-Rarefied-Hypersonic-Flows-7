# Physics-Guided-ML-for-Optimized-Grid-Generation-in-Rarefied-Hypersonic-Flows-7
A boundary condition–based machine learning (BCML) framework is developed for rarefied hypersonic flow-field prediction and mesh generation around re-entry bluff bodies. Trained on DSMC data, BCML predicts flow properties and generates mean-free-path-based meshes, reducing cell count and computational cost while maintaining accuracy."


Procedure for BCML Flow-Field Prediction and Adaptive Mesh Generation for Hypersonic Flow over a Circular Cylinder
    1. DSMC Dataset Generation
       A comprehensive DSMC database was first generated, consisting of 144 input features stored in a single file (Input_All_NR.txt) and four output flow-field variables stored in a separate file (Output_AllNR.txt). Owing to the large dataset size (approximately 13 GB for inputs and 2 GB for outputs), the complete training dataset is not included with the manuscript.
    2. BCML Flow-Field Model Training
       The provided training script (FF_BCML_Main_Training_Code.py) was executed on an Intel-based workstation to train the BCML flow-field prediction model. The training process resulted in the final model file (FF_BCML_Model_1502A.keras).
    3. Flow-Field Prediction and Validation
       Flow-field prediction was performed using the validation script (FF_BCML_Validation_Circle.py). The script utilizes the validation input dataset (InputFileWith16Generators_FF_CircleValidation) together with the trained BCML model (FF_BCML_Model_1502A.keras) to predict the primary flow-field quantities, namely pressure (P), translational temperature (T), and velocity (U).
    4. Flow-Field Post-Processing and Mean Free Path Calculation
       The predicted BCML flow-field data and corresponding DSMC reference data were imported into ParaView for visualization and analysis. Subsequently, cell-wise values of pressure (P), translational temperature (T), and velocity (U) were extracted throughout the computational domain. The local mean free path (MFP) was then calculated using the hard-sphere molecular model.
    5. Mean Free Path Validation­
       The script (lambda_plotting.py) was employed to validate the computed MFP distributions. The script takes as inputs the grid-cell center coordinates (C1), DSMC pressure and temperature fields, and BCML-predicted pressure and temperature fields. Comparative MFP contours were generated to assess the accuracy of the BCML predictions prior to mesh generation.
    6. Adaptive Mesh Generation
       Following successful MFP validation, the Python-based meshing framework (Meshing_code.py) was executed. The script uses the molecular number-density field (numb_density_data.txt) as input and generates the adaptive mesh file (bird_condition_mesh.msh) as output. The generated mesh can subsequently be imported into ParaView through the OpenFOAM polyMesh structure for visualization, mesh-quality assessment, and further analysis of mesh metrics.
