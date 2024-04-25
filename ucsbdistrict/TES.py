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
    CUP_instance : CUP = None  # Default value None for Cup instance
    HP_heatingCapacity: float 
    thermoclineDischarge: float 
    stopCharging: float 
    totalHeating_load : pd.Series 
    HW_districtSTP : pd.Series 
    districtHWRT : pd.Series 
    hotMaxflow: float
    coldMaxflow: float
    CUP_output_df : pd.DataFrame =None
    HW_upperTankTemp: float 
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
            "TES Hot Capacity hr-1 (%)": self.hotCapacityHr,
            "TES Hot Th after losses (Gal)_supply temp": self.hotThafterLoss_supply,
            "TES Hot Tc after losses (Gal)_return temp":self.hotThafterLoss_return,
            "TES Hot Thermocline (Gal)": self.hotThermocline,
            "TES Cold Th after losses (Gal)_supply temp": self.coldThafterLoss_supply,
            "TES Cold Tc after losses (Gal)_return temp":self.coldThafterLoss_return,
            "TES Cold Thermocline (Gal)": self.coldThermocline,

        })
        return df
    

    # @computed_field(return_type=MySeries)   
    # @property
    # def CUP_output_df(self):
    #     result_series = self.CUP_instance.compute_CUP()
    #     print("result_series",result_series)
    #     return result_series




    @computed_field(return_type=MySeries)   
    @property
    def hotStatus(self):
        # CUP_output_df = self.CUP_instance.compute_CUP()
        # print("CUP_output_df",CUP_output_df)
        # Apply conditions using pandas logical operations without direct if statements
        condition_hold = (self.CUP_output_df["Hot Load Shift Charging"] & ((self.hotThafterLoss_supply + self.hotThermocline) * self.thermoclineDischarge >= self.HW_TankVol * self.stopCharging) | (self.totalHeating_load > self.HP_heatingCapacity))
        condition_charge = self.CUP_output_df["Hot Load Shift Charging"] & ~condition_hold
        
        # Use numpy.select to assign results based on conditions
        result_series = pd.Series(pd.NA, index=self.CUP_output_df.index)  # Initialize result series with NA
        result_series[condition_hold] = "Hold"
        result_series[condition_charge] = "Charge"
        result_series[~self.CUP_output_df["Hot Load Shift Charging"]] = "Discharge"
        
        return result_series
        # return self.CUP_output_df["Hot Load Shift Charging"]


    @computed_field(return_type=MySeries)   
    @property
    def hotLoadShiftChargeRate(self):
        mask_OG =  (self.hotStatus == "Discharge") | (self.hotStatus == "Hold")

        HW_STP = (self.HW_districtSTP-self.districtHWRT)
        value1 = (self.HP_heatingCapacity-self.totalHeating_load)/500/HW_STP

        ##value2
        predictedHeatingMinusShift = self.CUP_output_df["Predicted Day's Heating Load (Btu/h)"]-self.CUP_output_df["Predicted Heating Load in Load Shift Window (Btu/h)"]
        mask = self.CUP_output_df["Hot Load Shift Charging"] !=0
        #initiate empty series with same index
        value2 = pd.Series(0, index=self.CUP_output_df.index)
        # #apply condition when mask is TRue(not zero)
        # value2[mask] =  predictedHeatingMinusShift/500/HW_STP/self.CUP_output_df["Hot Load Shift Hours"]

        # Compute the value for value2 and ensure dtype compatibility
        computed_value = predictedHeatingMinusShift/500/HW_STP/self.CUP_output_df["Hot Load Shift Hours"]
        value2[mask] = computed_value.astype(int)  # Convert to int64 dtype


        ##value 3
        value3= self.hotThermocline+self.hotThafterLoss_return*self.thermoclineDischarge

        ##value 4
        value4 = self.hotMaxflow
        print("v1",value1)
        print("v2",value2)
        print("v3",value3)
      
 
        result_series = pd.Series([0]*8760)
        result_series[mask_OG]=0
        # Filter and compute minimum across valid rows
        # valid_values = [value[~mask_OG] for value in [value1, value2, value3, value4]]
        # result_series[~mask_OG] = np.minimum.reduce(valid_values)

        valid_values = [value1, value2, value3, value4]
        valid_series = [value[~mask_OG] for value in valid_values if isinstance(value, pd.Series)]
        
        if valid_series:
            min_values = np.minimum.reduce(valid_series)
            # Ensure `min_values` has a compatible dtype with `result_series`
            min_values = min_values.astype(result_series.dtype)
            result_series[~mask_OG] = min_values
        
        
        return result_series



    @computed_field(return_type=MySeries)   
    @property
    def H_flowIntoTES(self):
        return np.where(self.hotStatus== "Charge", self.hotLoadShiftChargeRate,0)

            
    @computed_field(return_type=MySeries)   
    @property
    def H_tempTES(self):
        result_series = pd.Series([self.HW_upperTankTemp]*8760)
        return result_series
    
    @computed_field(return_type=MySeries)   
    @property
    def TES_H_flow(self):
        return self.totalHeating_load/500/(self.H_tempTES-self.districtHWRT).astype(float)




    @computed_field(return_type=MySeries)   
    @property
    def H_flowOutTES(self):
        minimum = np.minimum(self.flowTES,self.hotMaxflow,(self.hotThafterLoss_supply/60).astype(float)).astype(float)
        return np.where(self.hotStatus =="Discharge",minimum,0).astype(float)



    @computed_field(return_type=MySeries)   
    @property
    def TES_H_tc(self):
        value1 = self.hotThafterLoss_supply+self.H_flowIntoTES*60
        value2 = np.where(self.H_flowOutTES*60>self.hotThafterLoss_supply,self.hotThafterLoss_supply,self.H_flowOutTES*60)
        return value1 - value2


    @computed_field(return_type=MySeries)   
    @property
    def TES_H_capacity(self):
        try:
            result_series = self.TES_H_tc/self.HW_TankVol
        except ValueError:
            result_series = 0
        return result_series


    def compute_2(self):
        # return self.hotStatus
        df2 = pd.DataFrame({
            "TES Hot Status": self.hotStatus,
            "Hot Load-Shift Charge Rate (gpm)":self.hotLoadShiftChargeRate,
            "Flow into TES Hot Th (gpm)":self.H_flowIntoTES,
            "TES Hot Th Previous Hour (°F)":self.H_tempTES,
            "TES Flow to meet Campus Heating (gpm)":self.TES_H_flow,
            "Flow out of TES Hot Th (gpm)" : self.H_flowOutTES,
            "TES Hot Th (Gal)": self.TES_H_tc,
            "TES_H_tc" : self.TES_H_capacity
        })                
        return df2
