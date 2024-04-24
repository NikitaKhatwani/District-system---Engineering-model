from typing import Union
from pydantic import BaseModel, computed_field, validator
import pandas as pd
import numpy as np



class heatPump(BaseModel):
    districtHWRT : pd.Series
    districtCHWRT : pd.Series
    HW_districtSTP : pd.Series
    HP_leavingHW : pd.Series
    HP_enteringHW : pd.Series
    HP_enteringCHW : pd.Series
    HP_leavingCHW : pd.Series
    HP_maxHeating : pd.Series
    HP_maxCooling : pd.Series

    

    class MySeries(BaseModel):
        data: dict[str, Union[int, float, str]]  # Dictionary to store Series data

        @validator('data')
        def validate_data(cls, value):
            # Add validation logic here if needed (e.g., check data types, keys)
            return value

    class Config:
        arbitrary_types_allowed = True




    def min_diff(self, data: pd.Series, target: pd.Series) -> int:
        # Convert data_series and target_series to NumPy arrays for numerical operations
        data_values = data.values
        target_values = target.values
        # data_values = self.HP_enteringHW.values
        # target_values = self.districtHWRT.values
        
        # Reshape self.data to (1, 4) to broadcast across self.target
        data_values_reshaped = data_values.reshape(1, -1)  # Shape (1, 4)

        # Subtract each element of data_values_reshaped from every element of self.target
        diff = abs(target_values[:, np.newaxis] - data_values_reshaped)  # Broadcasting happens here
        # diff now has shape (8760, 4), where each row contains the differences

        # # Find the index of the minimum absolute difference for each element in self.target
        # min_index = np.argmin(np.abs(diff), axis=1)
        # # min_index now has shape (8760,) containing indices of minimum absolute differences
        # min_index = pd.Series(min_index)  
        return (diff,np.min(diff,axis=1))


    @computed_field(return_type=MySeries)
    @property 
    def CHWST(self):
        diff_HW_HWRT,min_diff_HW_HWRT =  self.min_diff(self.HP_enteringHW,self.districtHWRT)
        diff_CHW_CHWRT,min_diff_CHW_CHWRT =  self.min_diff(self.HP_enteringCHW,self.districtCHWRT)
        # reshaped_HP_leavingHW = self.HP_leavingHW.values.reshape((1, self.HP_leavingHW.size))
        # # Assuming self.HW_districtSTP is a pandas Series
        # rereshaped_HP_leavingHW = np.tile(reshaped_HP_leavingHW, (8760, 1))
        #reshape to convert single row to multiple rows to be able to broadcast
        rereshaped_HP_leavingHW = self.HP_leavingHW.values[np.newaxis, :].repeat(8760, axis=0)
        HW_districtSTP =self.HW_districtSTP.values
        HW_districtSTP = self.HW_districtSTP.values.reshape((-1, 1))
        matching_indices = np.where((diff_HW_HWRT==min_diff_HW_HWRT[:,np.newaxis])&(diff_CHW_CHWRT==min_diff_CHW_CHWRT[:,np.newaxis])& (rereshaped_HP_leavingHW == HW_districtSTP))[1]
        # Get the corresponding value from AE based on the matching index
        final_value = self.HP_leavingCHW[matching_indices] if matching_indices.size > 0 else None
        return final_value


    @computed_field(return_type=MySeries)
    @property 
    def heatingOutput(self):
        #get difference and min difference from the function
        diff_HW_HWRT,min_diff_HW_HWRT =  self.min_diff(self.HP_enteringHW,self.districtHWRT)
        diff_CHW_CHWRT,min_diff_CHW_CHWRT =  self.min_diff(self.HP_enteringCHW,self.districtCHWRT)
        #reshape to first convert single row with values then to multiple rows to be able to broadcast
        rereshaped_HP_leavingHW = self.HP_leavingHW.values[np.newaxis, :].repeat(8760, axis=0)
        HW_districtSTP = self.HW_districtSTP.values.reshape((-1, 1))
        matching_indices = np.where((diff_HW_HWRT==min_diff_HW_HWRT[:,np.newaxis])&(diff_CHW_CHWRT==min_diff_CHW_CHWRT[:,np.newaxis])& (rereshaped_HP_leavingHW == HW_districtSTP))[1]
        final_value = self.HP_maxHeating[matching_indices] if matching_indices.size > 0 else None
        return final_value
    
    @computed_field(return_type=MySeries)
    @property 
    def coolingOutput(self):
        diff_HW_HWRT,min_diff_HW_HWRT =  self.min_diff(self.HP_enteringHW,self.districtHWRT)
        diff_CHW_CHWRT,min_diff_CHW_CHWRT =  self.min_diff(self.HP_enteringCHW,self.districtCHWRT)
        #reshape to convert single row to multiple rows to be able to broadcast
        rereshaped_HP_leavingHW = self.HP_leavingHW.values[np.newaxis, :].repeat(8760, axis=0)
        HW_districtSTP = self.HW_districtSTP.values.reshape((-1, 1))
        matching_indices = np.where((diff_HW_HWRT==min_diff_HW_HWRT[:,np.newaxis])&(diff_CHW_CHWRT==min_diff_CHW_CHWRT[:,np.newaxis])& (rereshaped_HP_leavingHW == HW_districtSTP))[1]
        final_value = self.HP_maxCooling[matching_indices] if matching_indices.size > 0 else None
        return final_value
    
    @computed_field(return_type=MySeries)
    @property 
    def heatCoolRatio(self):
        print("ppp",self.coolingOutput)
        return self.heatingOutput/self.coolingOutput
    



    def compute(self):

        df = pd.DataFrame({
            "HP CHWST (°F)" : self.CHWST.reset_index(drop=True),
            "Max Unitary Heating Output (Btu/h)": self.heatingOutput.reset_index(drop=True),
            "Max Unitary Cooling Output (Btu/h)":self.coolingOutput.reset_index(drop=True),
            "Heat:Cool": self.heatCoolRatio.reset_index(drop=True),

        })

        return df


    