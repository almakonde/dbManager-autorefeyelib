# scr run AutoRefGUI
import sys
import os
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        print(f'delteting {kIdx}')
        del sys.modules[kIdx]
# import AutoRefGUI
from autorefeyelib.GUI.AutoRefGUI import GUI
# pid = os.fork()

# data_path = '/home/ofir/Work/EyeLib/Data/vxSampleFiles/'
data_path = os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','SampleData','vx120')# '/home/ofir/Work/EyeLib/Data/SampleVXZipFiles'
with GUI(data_path, monitorVxFolder=True) as g:
    sys.exit(g.gui_main_window.exec())
