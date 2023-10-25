# Overview
This package contains libraries to predict IOL power, refraction and various risk factors from objective measurements.
For convenience, it allows a single call to run all ML related computation of the eyelib

The main pipeline (currently) runs with two inputs:
1. the vx120/vx130 zip file path (result file from a single examination)
2. the revo output bmp file path

#  Dependencies
Install dependencies for the whole package using:
```python
pip install -r Requirements/All.txt
```

# Demo
For a demo run:
```python
python scrDemo.py
```
from the script folder (project folder).
The script uses the data in the 'SampleData' folder, to produce results DataFrame
with parsed and transformed values ready for display, and all predicted values


# Quick start
From the project folder, use the EyelibMain pipeline to run the whole process as follows:
```python
from autorefeyelib.Pipline import EyelibMain
pipe = EyelibMain.Pipeline()
refPred, iolPred = pipe.Run(vx120ZipFilePath,revoBmpFilePath)
```
[Outputs](README.md#output) include DataFrames for predicted refraction and predicted IOL power.

# Sample input data
Sample vx120 zip files Revo OCT bmp files can be found in `autorefeyelib/SampleData`

# Example
From the project folder
```
from autorefeyelib.Pipline import EyelibMain
import os
revoImageFile = os.path.join(os.path.dirname(__file__),'autorefeyelib','SampleData','Revo','sample02.bmp')
vx120zipFile  = os.path.join(os.path.dirname(__file__),'autorefeyelib','SampleData','vx120','sample02.zip')
pipe = EyelibMain.Pipeline()
refPred, iolPred = pipe.Run(vx120zipFile, revoImageFile)
```
The output `refPred` (see [Output](README.md#output)) is a DataFrame consisting of parsed and transformed vx120 measurements, parsed OCT data, and risks calculated (see [Predicted risks](README.md#predicted-risks))
The output `iolPred` is an array of DataFrame, corresponding to the results of IOL prediction for each A-constant (see [Predicted IOL](README.md#predicted-iol))

### Optional parameters:
1. `patID` -(str/int) patient ID can be passed into the pipline as a string or an integer, to retrieve a dataframe output with an index set as the patientID
     ```
     refPred, iolPred = pipe.Run(vx120ZipFile,revoBMPOfilePath,patID='patient01')
     ```
     the resulted `refPred` DataFrame will have `index=patient01`
2. `vertexDistance`- (float) the vertex distance in m units used to transform refraction values from the cornea plane to the spectable plane. Default=0.012m
3. `Aconst` (array of float)- the IOL manufacturer A-constant for a chosen lens. Default=[118.9] (unitless), example [118.5, 118.9, 119.4]
4. `targetRefraction` (array of float)- target refractions (Diopter) to desired refraction after IOL implantation. Default=[0], example [-1, 0, -2]


# Main pipeline
Input and output schematic is presented in the Figure below:
![Main pipeline Input/Output](/Documents/PipelineMain_IO.png)

# Output
The `refPred` output (DataFrame) will contain concatenated measurements and predicted values in a single row with an index=PatientID.
All predicted values field names start with `Predicted_` as listed below for left or right eye. The reminder of the fields represent parsed and transformed data
from the vx120 and Revo. Content of predicted fields appear below.

### Predicted refraction
Output fields related to refraction contant

| Predicted (units)| Field name in output |
| ------   | ------ |
| Sphere (D)   | `Predicted_Sphere_Right` or `_Left` |
| Cylinder (D) | `Predicted_Cylinder_Right` or `_Left` |
| Axis    (deg) | `Predicted_Axis_Right` or `_Left` |
| addition (D) | `Predicted_Add_Right` or `_Left` |
| contact-lens sphere  (D) | `Predicted_Contact_Sphere_Right` or `_Left`|
| contact-lens cylinder (D) | `Predicted_Contact_Cylinder_Right` or `_Left`|
| contact-lens axis     (deg) | `Predicted_Contact_Axis_Right` or `_Left`|


### Predicted risks
| Risk | Field name in output |
| ------ | ------ |
| Tonometry     | `Predicted_Risk_Tono_Right` or `_Left` |
| Pachymetry    | `Predicted_Risk_Pachy_Right` or `_Left` |
| Keratoconus   | `Predicted_Risk_Keratoconus_Right` or `_Left` |
| Angle closure | `Predicted_Risk_AngleClosure_Right` or `_Left`|

the output predicted risk are strings of either: Low Medium or High

### Predicted IOL
| Predicted (units) | Field name in output |
| ------ | ------ |
| IOL power (D)   | `Predicted_IOL_Power_Right` or `_Left` |
| Final refraction (D)  | `Predicted_IOL_Refraction_Right` or `_Left` |
| IOl formula     | `Predicted_IOL_Formula_Right` or `_Left` |

The Predicted IOL formula is a string.

### Parsed OCT (Revo) measurements
The Revo OCT image is used as an input, and OCR is performed on the image to extract values from 10 successive measurements, as well as the mean.
The measurement values extracted will appear in the ourput DataFrame `refPred` (See [Example](README.md#example)) with the following field names:
+ lens thickness `LT_<i>_Right` or `_Left` (in mm)
+ central cornea thickness `CCT_<i>_Right` or `_Left` (in mm)
+ axial length `Axial_Length_<i>_Right` or `_Left` (in mm)
+ anterior chamber depth `ACD_<i>_Right` or `_Left` (in mm)

where `<i>` is an integer 0-9, to represent each of the 10 measurements,  or `<i>` could be `Avg`, to represent the average of **valid** measurements

### Transformed values
Parsed measurements undergo transformation before passed to the various predictors, and are made ready for display in the final report.
Prior to transformation, the parsed DataFrame undergo imputation, to assure no values are missing.
Various measurements such as  sph/cyl are rounded to 2 decimel digits in the process of transformation.
In addition, several computation are performed using the objective parsed measurements:
1. anterior chamber volume;
2. maximal keratometry
3. patient age at time of examination
4. corrected intra-ocular pressure
5. corneal radius (translation from diopters to mm)
6. translation of keratometry from mm to diopters
7. patient ID assignment based on firstname+surname+date-of-birth
8. High order aberrations (HOA)
9. Low order aberrations (LOA)



