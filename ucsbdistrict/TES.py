from typing import Union
from pydantic import BaseModel, computed_field, validator
import pandas as pd
import numpy as np
from ucsbdistrict.CUP import CUP

class TES(BaseModel):
    # coolingLoad: pd.Series
    # heatingLoad : pd.Series
    HW_upperTankVol : float
    HW_lowerTankVol : float
    HW_TankVol : float
    HW_wallLoss : float
    HW_ThermocloneLoss : float
    CHW_upperTankVol : float
    CHW_lowerTankVol : float
    CHW_TankVol : float
    CHW_wallLoss : float
    CHW_ThermocloneLoss : float
    CUP_instance : CUP = None  # Default value None for address
    HP_heatingCapacity: float
    thermoclineDischarge: float
    stopCharging: float
    totalHeating_load : pd.Series
    # hotLoadShiftCharging : CUP.HW_charging

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
    def hotCapacityHr(self):

        # Calculate the value of interest (self.HW_upperTankVol / self.HW_TankVol)
        value = self.HW_upperTankVol / self.HW_TankVol

        # Repeat the calculated value 8760 times to create a Series
        series_values = np.repeat(value, 8760)

        # Create a pandas Series using the repeated values
        series = pd.Series(series_values)

        return series
    
    def tempAfterLosses(self,capacity,tankVol,wallLoss,thermocloneLoss):
        condition = (capacity >= 0.75) | (capacity <= 0.25)

        # Calculate values based on condition using np.where()
        value_if_true = tankVol - (tankVol * wallLoss / 100)
        value_if_false = tankVol - (tankVol * (wallLoss + thermocloneLoss) / 100)

        # Use np.where() to apply the condition element-wise and compute the resulting Series
        result_series = np.where(condition, value_if_true, value_if_false)

        # Return the resulting Series as a MySeries instance
        return pd.Series(result_series)

    @computed_field(return_type=MySeries)   
    @property
    def hotThafterLoss_supply(self):
        result = self.tempAfterLosses(self.hotCapacityHr,self.HW_upperTankVol,self.HW_wallLoss,self.HW_ThermocloneLoss)
        return result

    

    
    @computed_field(return_type=MySeries)   
    @property
    def hotThafterLoss_return(self):
        result = self.tempAfterLosses(self.hotCapacityHr,self.HW_lowerTankVol,self.HW_wallLoss,self.HW_ThermocloneLoss)
        return result
    
    @computed_field(return_type=MySeries)   
    @property
    def hotThermocline(self):
        result = self.HW_TankVol-self.hotThafterLoss_supply-self.hotThafterLoss_return
        return result


    @computed_field(return_type=MySeries)   
    @property
    def coldCapacityHr(self):

        # Calculate the value of interest (self.HW_upperTankVol / self.HW_TankVol)
        value = self.CHW_upperTankVol / self.CHW_TankVol

        # Repeat the calculated value 8760 times to create a Series
        series_values = np.repeat(value, 8760)

        # Create a pandas Series using the repeated values
        series = pd.Series(series_values)

        return series
    
    @computed_field(return_type=MySeries)   
    @property
    def coldThafterLoss_supply(self):
        result = self.tempAfterLosses(self.coldCapacityHr,self.CHW_upperTankVol,self.CHW_wallLoss,self.CHW_ThermocloneLoss)
        return result
    
    @computed_field(return_type=MySeries)   
    @property
    def coldThafterLoss_return(self):
        result = self.tempAfterLosses(self.coldCapacityHr,self.CHW_lowerTankVol,self.CHW_wallLoss,self.CHW_ThermocloneLoss)
        return result
    
    @computed_field(return_type=MySeries)   
    @property
    def coldThermocline(self):
        result = self.CHW_TankVol-self.coldThafterLoss_supply-self.coldThafterLoss_return
        return result



    def compute(self):
        df = pd.DataFrame({
            "check": self.hotCapacityHr,
            "TES Hot Th after losses (Gal)_supply temp": self.hotThafterLoss_supply,
            "TES Hot Tc after losses (Gal)_return temp":self.hotThafterLoss_return,
            "TES Hot Thermocline (Gal)": self.hotThermocline,
            "TES Cold Th after losses (Gal)_supply temp": self.coldThafterLoss_supply,
            "TES Cold Tc after losses (Gal)_return temp":self.coldThafterLoss_return,
            "TES Cold Thermocline (Gal)": self.coldThermocline,

        })
        return df
    
    @computed_field(return_type=MySeries)   
    @property
    def hotStatus(self):
        CUP_output_df = self.CUP_instance.compute_CUP()
        
        # Apply conditions using pandas logical operations without direct if statements
        condition_hold = (CUP_output_df["Hot Load Shift Charging"] & ((self.hotThafterLoss_supply + self.hotThermocline) * self.thermoclineDischarge >= self.HW_TankVol * self.stopCharging) | (self.totalHeating_load > self.HP_heatingCapacity))
        condition_charge = CUP_output_df["Hot Load Shift Charging"] & ~condition_hold
        
        # Use numpy.select to assign results based on conditions
        result_series = pd.Series(pd.NA, index=CUP_output_df.index)  # Initialize result series with NA
        result_series[condition_hold] = "Hold"
        result_series[condition_charge] = "Charge"
        result_series[~CUP_output_df["Hot Load Shift Charging"]] = "Discharge"
        
        return result_series





    def compute_2(self):
        # return self.hotStatus
        df2 = pd.DataFrame({
            "TES Hot Status": self.hotStatus
        })                
        return df2
