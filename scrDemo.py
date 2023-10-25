import os
import sys
import time
# Remove previously loaded autoref modules (DEBUG mode)
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'Deleting {kIdx}')
        del sys.modules[kIdx]

from autorefeyelib.Pipeline import EyelibMain

"""
  A simple script to demonstrate prediction of subjective refraction from objective data
  and the IOLpower


  Parameters:
  -----------
  vx120zipFile, str
    a path to the vx120 zip file containined patient objective measurement
  revoImageFile, str
    a path to the revo bmp output image containing biometry
    if no bmp file is available, set None
    in that case, values used for prediction will be averages (from IOLmaster)
  patID, str, default='0'
     patient ID passed to the pipeline. resulted prediction DataFrame will
     have the patID as the index
  vertexDistance, float, default=0.012
     the vertex distance (m)
  Aconst, float, default=118.9
    manufacturer A-constant for the IOL
  targetRefraction, float, default=0
    target refraction for the implantation of IO:
    set to emmatropia by default

  Output:
  ------

  refPred, DataFrame
    Predicted subjective refraction for left and right eye, for spectacles and contact lenses
  risks
    Predicted risks of angle closure, keratoconus, pachy, and tono, based on objective measurements
  iolPred, list(DataFrame),
    predicted IOL power refraction and formula for each values of the A constant
    Each element of the list of DFs correspond to  a value of the target refraction input
    e.g. Predicted_Sphere_Right,
        Predicted_Contact_Cylinder_Left
  recommendedSurgery, dict
    dictionary of recommended refractive surgery for right and left eye, respectively
  transData, DataFrame
    transformed vx120data and concatenated revo data after parsing

  Notes:
  ------
  * All predicted values in the dataframe header have prefix of Predicted_
  This function is for demonstration of retrieval of subjective refraction
  from objective vx120 measurements, prediction of various risks and the prediction of IOL power.
  This function is not optimal in terms of running time.
  It loads data such as csv and deserializes models which results in
  long loading time. A typical running time of this function is 4-6 sec.

  A more optimal use of the classes below would be to daemonize the parser and the predictor
  to be loaded once at startup and wait for input values

"""
#______ Parameters________
startTime     = time.time()
xmlResFolder  = os.path.join(os.path.dirname(__file__), 'Results')
revoImageFile = os.path.join(os.path.dirname(__file__),'autorefeyelib','SampleData','Revo','sample02.bmp')
vx120zipFile  = os.path.join(os.path.dirname(__file__),'autorefeyelib','SampleData','vx120','sample02.zip')
vx40File      = os.path.join(os.path.dirname(__file__),'autorefeyelib','SampleData','vx40','sample02.xml')
virtual_agent_file = None
#_________________________

#____Get predictions______
p                  = EyelibMain.Pipeline()
refPred,risks,iolPred,recommendedSurgery,transData = p.Run(vx120zipFile,revoImageFile,
                    vx40File=vx40File,
                    virtualAgentFile=virtual_agent_file,
                    patID=44,
                    vertexDistance=0.012,
                    Aconst=[118,118.5,118.9,119.4],
                    targetRefractionIOL=[0,-0.5,-1.0,-1.5],
                    targetRefractionSurg=(0,0),
                    pvd = (False,False),
                    dominantEye='right',
                    priorLasik = (False,False),
                    flapThickness=(100,100)
                    )

#____Optional output_______
# Export a vx40 xml with populated predicted values
# # Uncomment the lines below to export  a vx40 xml with predicted refraction values
# os.makedirs(xmlResFolder,exist_ok=True)
# p.refractionPredictor.ExportPredictionToVx40xml(xmlResFolder,refPred)

print(f"Running time   : {time.time()-startTime:.2f} sec")

print("\n___Refraction_____\n")
ind = refPred.index[0]
print(f"Objective                           :{transData['WF_SPHERE_R_3_Right'][ind]:.2f}\
({transData['WF_CYLINDER_R_3_Right'][ind]:.2f}){transData['WF_AXIS_R_3_Right'][ind]:.0f} /\
{transData['WF_SPHERE_R_3_Left'][ind]:.2f}({transData['WF_CYLINDER_R_3_Left'][ind]:.2f}){transData['WF_AXIS_R_3_Left'][ind]:.0f}")
print(f"Predicted Subjective spectacles     :{p.refractionPredictor.FormatPredictionAsString(refPred,rtype='spectacles')}")
print(f"Predicted subjective contacts lenses:{p.refractionPredictor.FormatPredictionAsString(refPred,rtype='contacts')}")

print("\n_________ IOL power prediction _______\n")
print("         |       Right     |         Left   |")
print("____________________________________________")
print("A const. | power | formula | power  | formula ")
print("____________________________________________")
# display results of IOL prediction
for kIdx in range(len(iolPred)):
  for rIdx in range(len(iolPred[kIdx])):
    iolRes = iolPred[kIdx].iloc[rIdx]
    p_right = iolRes[f"Predicted_IOL_Power_Right"]
    p_left  =iolRes[f"Predicted_IOL_Power_Left"]
    f_right = iolRes[f"Predicted_IOL_Formula_Right"]
    f_left  = iolRes[f"Predicted_IOL_Formula_Left"]
    print(f'{iolRes["Aconst"]}    | {p_right}  | {f_right}    | {p_left}   | {f_left}')


# display results of the surgery recommender
print("\n______ Surgery Recommender ___________\n")
for eIdx in ['Right', 'Left']:
  print(f"\nRecommendation for refraction surgery {eIdx} eye:")
  print("____________________________________________________\n")
  for kIdx in recommendedSurgery[eIdx]['Decision'].keys():
      print(f"{kIdx}: {recommendedSurgery[eIdx]['Decision'][kIdx]}")
