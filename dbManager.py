#import for file path handler
import os

#import for DataFrames

import pandas as pd
#import Parsers
from autorefeyelib.Parsers import vx120
from autorefeyelib.Parsers import Revo
from autorefeyelib.Parsers import xmlParser
from autorefeyelib.Parsers import PrevotEMRParser

MAXX = 30000
pd.options.display.max_rows = MAXX

#patientID comes from business logic authorisation session
#files comes from the same session
#input: files, patientINFO, devices list
#check if we have the same patients, if there are no patients
#TODO: move functions out of file , or put them inside the class

class DBManager:
	""" merging the data into the dict() of the DataFrames()"""
	def __init__(self):
		# output dict()
		self.db              = dict()
		self._deviceList     = ['vx120', 'vx40', 'revo', 'virtual_agent']
		for kIdx in self._deviceList:
			self.db[kIdx] = pd.DataFrame()

		self._parsers = {'vx40':auto_parse_vx40(),
						'vx120':auto_parse_vx120(),
						'virtual_agent': auto_parse_virtual_agent(),
						'revo':auto_parse_revo}

	def _SetDevices(self, deviceTypes:list):
		"""  A service function to set new devices for the dbManager"""
		self._deviceINFO = deviceTypes
		# TODO: check if keys already exist and warn not to overwrite values
		for kIdx in self._deviceINFO:
			self.out[kIdx] = None

	def Get(self,patID :int, dType: str)->dict():
		# pd.options.display.max_rows = 100
		"""
		 dType: str,
		  vx40,
		  vx120
		  revo
		  virtualAgent
		  all

		 Returns
		 --------
		 data_out: dictionary
		  output dictionary
		"""
		# check if deviceName appears in the list of data available in the class
		data_out = {}
		if not isinstance(dType,str):
			print(f'dType must be a string. Got {dType.__class__}')
		if dType.lower()=='all':
			# return all data of a certain patient
			# TODO: here you have to build the output structure (dictionary )
			for dIdx in self._deviceList:
				data_out[dIdx] = self.db[dIdx][patID]
		else:
			for dIdx in dType:
				data_out[dIdx]= self.db[dIdx][patID]

		return data_out

	def Parse(self, file_path, deviceType):
		"""
		don't forget docstrings

		Parameters
		----------

		output
		---------


		 dont forget to check input types
		"""
		if not isinstance(file_path,str):
			raise ValueError(f'file_path must be string. got {file_path.__class__}')
		else:
			return self._parsers[deviceType](file_path)


	def _auto_parse_vx120(_file_path=None):
		vx120zipFile  = os.path.dirname(_file_path)
		vx120Parser = vx120.Parser()
		vx120_data = vx120Parser.ParseVXFromZip(vx120zipFile)
		return vx120_data

	def _auto_parse_vx40(_file_path=None):
		vx40File      = os.path.dirname(_file_path)
		vx40Parser = xmlParser.Parser()
		vx40_data = vx40Parser.Parse(vx40File)
		return vx40_data.T

	def _auto_parse_revo(_file_path=None):
		revoImageFile = os.path.dirname(_file_path)
		revoParser = Revo.Parser()
		revo_data = revoParser.Parse(revoImageFile)
		return revo_data

	def _auto_parse_virtual_agent(_file_path=None):
		virtual_agent_file = pd.DataFrame()
		virtual_agent_data = None
		#parsing...
		return virtual_agent_data

	def _register_parser(self, parser_name:str, parser_call):
		""" register a new parser (device) and its associated method call"""
		for kIdx in _parser:
			if _parser[parser_name] == parser_name:
				self._parser[parser_name]=parser_call
		

	def _remove_parser(self, parser_name):
		# remove a parser method call from list of parsers (device)
		# NOTE: do not remove the data. only the call
		pass

	def Set(self, data: pd.DataFrame, patID:int, deviceType:str, optional_arg=None)->int:
		"""
		don't forget doc strings

		Parameters
		-----------

		output
		---------

		"""
		# if the patient does not exist, create new patient and add the row(s)
		# self.CreateNewPatient(patID,...)

		# if the patient exist, append the row(s)
		#
		if not isinstance(data,(pd.DataFrame,pd.Series)):
			raise ValueError(f'input data should be of type pandas.DataFrame. got {data.__class__}')
		if not isinstance(deviceType,str):
			raise ValueError(f'deviceType should be of type string. got {deviceType.__class__}')

		#  NOTE: a patient might already exist in the database
		# we need to check where to add the data
		if len(self.out[deviceType].loc[patID])>0:
			# if the patient exist in the db for the particular device
			# append. You need to check that column names are correct and are entered in the right order
			# there is a function for that in pandas.
			pass
		else:
			# create new patient's device entry
			self.out[deviceType].loc[patID] = data
		# parsing input file

		return 0




     	#try:
     		#case(dType:str='vx120'):

     			#return self.out

     		#case(dType:str='vx40'):

     			#return self.out

     		#case(dType:str='revo'):

     			#return self.out

     		#case(dType:str='virtualagent'):


     		#case(dType:str='all'):


     	#catch()





#data = DBManager()
#data.Set(auto_parse_revo(), 0, 'revo')
#data.Get(0, 'revo')


def save_to_file(data):

	f = open("test.txt", "w")
	f.write(data)
	f.close()




#----------------------------------------------------------------------------test----------------------------------------------------
data_revo = auto_parse_revo()
data_vx120 = auto_parse_vx120()
data_vx40 = auto_parse_vx40()
print(data_vx40.index)
print(data_revo.index)
print(data_vx120.index)




out = dict()
out = {'revo': data_revo,
	   'vx120': data_vx120,
	   'vx40': data_vx40.T}
#print(out)
print(type(out))
#print('vx40index' + data_vx40.index + 'vx120index' + data_vx120.columns + 'revoindex' + data_revo.columns)
#print('vx40index' + data_vx40.key + 'vx120index' + data_vx120.key + 'revoindex' + data_revo.key)
#print('vx40index' + data_vx40.loc[0] + 'vx120index' + data_vx120.loc[0] + 'revoindex' + data_revo.loc[0])
save_to_file(str(out))
print('vx40')
print(data_vx40.index)
print('vx120')
print(data_vx120.index)
print('revo')
print(data_revo.index)



























