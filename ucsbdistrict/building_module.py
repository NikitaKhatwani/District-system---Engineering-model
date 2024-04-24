from typing import Union
from pydantic import BaseModel, computed_field, validator
import pandas as pd
import numpy as np
from ucsbdistrict.building_staticInputs import BuildingParameters

class Building(BaseModel):
    #adding changing inputs needed  for formulas
    parameters: BuildingParameters
    caan_no: int
    name: str
    coolingLoad: pd.Series
    heatingLoad : pd.Series
    DHWLoad : pd.Series
    timeStamp : pd.Series 
    CHW_maxLoad : float
    DHW_maxLoad : float
    DHW_loadMinApproach : float
    


    class MySeries(BaseModel):
        data: dict[str, Union[int, float, str]]  # Dictionary to store Series data

        @validator('data')
        def validate_data(cls, value):
            # Add validation logic here if needed (e.g., check data types, keys)
            return value

    class Config:
        arbitrary_types_allowed = True


    @computed_field(return_type=MySeries)
    @property
    def loopHWST(self) -> pd.Series:
        return self.parameters.HW_LoopSTP + self.parameters.HW_supplyLosses
    
    @computed_field(return_type=MySeries)
    @property
    def loopCHWST(self) -> pd.Series:
        return self.parameters.CHW_LoopSTP + self.parameters.CHW_supplyLosses
    
    @computed_field(return_type=MySeries)
    @property
    def CHWRT(self) -> pd.Series:
        return self.loopCHWST+((self.parameters.CHW_deltaT_Max-self.parameters.CHW_deltaT_Min)/self.CHW_maxLoad\
                              )*self.coolingLoad+self.parameters.CHW_deltaT_Min
    
    @computed_field(return_type=MySeries)
    @property
    def CHWRflow(self) -> pd.Series:
            return  self.coolingLoad / 500 / (self.CHWRT - self.loopCHWST)
    
    @computed_field(return_type=MySeries)
    @property 
    def min_index(self):
        # Convert data_series and target_series to NumPy arrays for numerical operations
        data_values = self.parameters.HHW_supply_Temps.values
        target_values = self.parameters.HHW_BldgSTP.values
        
        # Reshape self.data to (1, 4) to broadcast across self.target
        data_values_reshaped = data_values.reshape(1, 4)  # Shape (1, 4)

        # Subtract each element of data_values_reshaped from every element of self.target
        diff = target_values[:, np.newaxis] - data_values_reshaped  # Broadcasting happens here
        # diff now has shape (8760, 4), where each row contains the differences

        # Find the index of the minimum absolute difference for each element in self.target
        min_index = np.argmin(np.abs(diff), axis=1)
        # min_index now has shape (8760,) containing indices of minimum absolute differences
        min_index = pd.Series(min_index)  
        return min_index

    @computed_field(return_type=MySeries)
    @property 
    def HHWRT(self):
          try:
            # Create a new Series with sequential index and values from self.delta
            new_deltaSeries = pd.Series(self.parameters.HHW_return_Temps[self.min_index].values, index=range(len(self.min_index)))
            return self.parameters.HHW_BldgSTP-new_deltaSeries
          except IndexError:
            return None 

    @computed_field(return_type=MySeries)
    @property
    def HHWRflow(self) -> pd.Series:
            return  self.heatingLoad / 500 / (self.parameters.HHW_BldgSTP - self.HHWRT) 

    @computed_field(return_type=MySeries) 
    @property
    def DHWtemp(self):
        return self.parameters.DHWSetpoint[self.parameters.DHW_indices].reset_index(drop=True)
    
    @computed_field(return_type=MySeries)
    @property
    def DHWRT(self):
        DHWRT = self.DHWtemp + np.maximum(self.parameters.DHWminApproach, (self.parameters.DHWmaxApproach-self.parameters.DHWminApproach)/\
                                               (1-self.DHW_loadMinApproach/self.DHW_maxLoad)*self.DHWLoad/self.DHW_maxLoad\
                                               +self.parameters.DHWmaxApproach-(self.parameters.DHWmaxApproach-self.parameters.DHWminApproach)/\
                                               (1-self.DHW_loadMinApproach/self.DHW_maxLoad))
        return DHWRT
    
    @computed_field(return_type=MySeries)
    @property
    def DHWRflow(self) -> pd.Series:
            return  self.DHWLoad / 500 / (self.loopHWST - self.DHWRT)
    
    @computed_field(return_type=MySeries)
    @property
    def districtHWSflow(self):
        return self.DHWRflow +(self.HHWRflow*(self.parameters.HHW_BldgSTP-self.HHWRT))/(self.loopHWST-self.HHWRT)
    
    @computed_field(return_type=MySeries)
    @property
    def bypassHHWS(self):
        return self.HHWRflow-(self.districtHWSflow-self.DHWRflow)
    
    @computed_field(return_type=MySeries)
    @property
    def HWRflow(self):
        return self.HHWRflow-self.bypassHHWS+self.DHWRflow
    
    @computed_field(return_type=MySeries)
    @property
    def districtHWRT(self):
        return (self.HHWRT*(self.HHWRflow-self.bypassHHWS)+self.DHWRflow*self.DHWRT)/self.districtHWSflow
    
    @computed_field(return_type=MySeries)
    @property
    def HWSequalHWR(self):
        # Define tolerance for comparison
        tolerance = 1e-6

        return (abs(self.districtHWSflow - self.HWRflow) < tolerance)

    def compute(self):

        # Create a list of unique identifiers for each row (e.g., row numbers)
        index_list = range(len(self.loopHWST))  # Assuming you want to use row numbers

        df = pd.DataFrame({
            "caan_no":self.caan_no,
            "Time Stamp" : self.timeStamp,
            "Space Heating Load (Btu/h)":self.heatingLoad,
            "DHW Load (Btu/h)":self.DHWLoad,
            "Cooling Load (Btu/h)":self.coolingLoad,
            'Loop HWST @ Building (°F)': self.loopHWST,
            'Loop CHWST @ Building (°F)': self.loopCHWST,
            'CHWRT (°F)': self.CHWRT,
            "CHWR Flow (gpm)":self.CHWRflow,
            "Building HHWRT (°F)" : self.HHWRT,
            "HHWRflow": self.HHWRflow,
            "Building Domestic Water Temp (°F)": self.DHWtemp,
            "Building DHWRT (°F)": self.DHWRT,
            "Building DHWR Flow (gpm)": self.DHWRflow,
            "District HWS Flow (gpm)": self.districtHWSflow,
            "Bypassed Return to HHWS (gpm)": self.bypassHHWS,
            "District HWR Flow (gpm)":self.HWRflow,
            "District HWRT (°F)": self.districtHWRT,
            "Check Building HWS = HWR":self.HWSequalHWR
        }, index=index_list)
        
        
        
        

        # # Create the DataFrame with the index
        # df = pd.DataFrame(df_data, index=index_list)

        return df