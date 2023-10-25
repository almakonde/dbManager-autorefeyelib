import sys
import os
# Remove instances of the autorefeyelib classes
for kIdx in list(sys.modules.keys()):
    if kIdx.find('autorefeyelib')!=-1:
        del sys.modules[kIdx]
from autorefeyelib.Parsers.vx40 import Parser as vx40Parser

filePath  = os.path.join(os.path.dirname(__file__),'..','..','autorefeyelib','SampleData','vx40','sample01.xml')
xmlParser = vx40Parser()
xmlRoot   = xmlParser.Parse(filePath,index='ofir')