from typing import Union
from pydantic import BaseModel, computed_field, validator
import pandas as pd
import numpy as np

class BuildingParameters(BaseModel):
    # adding static inputs
    HW_LoopSTP : pd.Series
    HW_supplyLosses : int 
    CHW_LoopSTP : pd.Series
    CHW_supplyLosses : int  
    CHW_deltaT_Max : int  
    CHW_deltaT_Min : int  
    CHW_maxLoad : float
    HHW_supply_Temps :pd.Series
    HHW_return_Temps :pd.Series
    HHW_BldgSTP : pd.Series 
    DHWMonths : pd.Series 
    DHWSetpoint : pd.Series
    buildingDate : pd.Series
    hotWaterMonths : np.ndarray
    # DHW_indices : pd.Series
    DHWmaxLoad : float
    DHWmaxApproach : int  
    DHWminApproach : int  
    DHWloadMinApproach : float
    
            
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
    def DHW_indices(self) -> pd.Series:
        month_to_match = self.buildingDate.dt.month.values

        # Find the index where month matches
        DHW_indices = np.where(month_to_match[:, None] == self.hotWaterMonths )[1]

        # Convert NumPy array to Series
        DHW_indices = pd.Series(DHW_indices) 
        
        return DHW_indices