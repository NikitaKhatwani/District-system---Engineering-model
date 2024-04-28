from typing import Union
from pydantic import BaseModel, computed_field, validator
import pandas as pd
import numpy as np



class CUP(BaseModel):

    timeStamp : pd.DataFrame
    hotTank_schedule : pd.DataFrame
    coldTank_schedule : pd.DataFrame
    totalHeating_load : pd.Series
    totalCooling_load :  pd.Series


    

    class MySeries(BaseModel):
        data: dict[str, Union[int, float, str]]  # Dictionary to store Series data

        @validator('data')
        def validate_data(cls, value):
            # Add validation logic here if needed (e.g., check data types, keys)
            return value

    class Config:
        arbitrary_types_allowed = True


    

    def is_on(self,tank_schedule,month,hour):
        #loc is used to get the valu at the month and hour of the schedule using "labels- row or col name"
        value = tank_schedule.loc[tank_schedule["Month"]== month,hour]
        if  value.iloc[0]==1:
            return True
        else:
            return False 


    @computed_field(return_type=MySeries)
    @property    
    def HW_charging(self):
        df = pd.DataFrame()
        result = []
        for ts in self.timeStamp['Date and Time']:
            month = ts.month
            hour = ts.hour
            # print("ts",month, hour)
            result.append(self.is_on(self.hotTank_schedule, month, hour))
        # print(df)
        return pd.Series(result)
    

    @computed_field(return_type=MySeries)
    @property    
    def CHW_charging(self):

        result = []
        for ts in self.timeStamp['Date and Time']:
            month = ts.month
            hour = ts.hour

            result.append(self.is_on(self.coldTank_schedule, month, hour))
        # print(df)
        return pd.Series(result)
    


    @computed_field(return_type=MySeries)
    @property    
    def shiftCount(self):

        value = self.timeStamp['Date and Time'].dt.dayofyear-1
        return value
        

    def predictedDayLoad(self,shiftCount,load):
        data ={
            "shiftCount": shiftCount,
            "load": load
        }
        df = pd.DataFrame(data)

        #initiate empty series to store results
        result_series = pd.Series(index=shiftCount.index)

        for value in shiftCount.unique():
            
            #filter to every shiftcount value
            filtered_df = df[df["shiftCount"]==value]
            summed_value = filtered_df["load"].sum()

            result_series[shiftCount==value]=summed_value
        return result_series




    @computed_field(return_type=MySeries)
    @property    
    def predictedDay_heating(self):
        x= self.predictedDayLoad(self.shiftCount,self.totalHeating_load)
        return x
    

    @computed_field(return_type=MySeries)
    @property    
    def predictedDay_cooling(self):
        x= self.predictedDayLoad(self.shiftCount,self.totalCooling_load)
        return x


    def predictedDayLoad_shift(self,shiftCharging,shiftCount,load):
        data ={
            "shiftCharging" :shiftCharging,
            "shiftCount": shiftCount,
            "load": load
        }
        df = pd.DataFrame(data)

        #initiate empty series to store results
        result_series = pd.Series(index=shiftCount.index)

        #this is working but write better code to not use loop to assign value to every value
        for value in shiftCount.unique():
            filtered_df = df[(df["shiftCount"]==value) & (df["shiftCharging"]==True)]
            summed_value = filtered_df["load"].sum()
            result_series[shiftCount==value]=summed_value

        return result_series


    @computed_field(return_type=MySeries)
    @property    
    def predictedDay_heating_shift(self):
        x= self.predictedDayLoad_shift(self.HW_charging,self.shiftCount,self.totalHeating_load)
        return x
    
    @computed_field(return_type=MySeries)
    @property    
    def predictedDay_cooling_shift(self):
        x= self.predictedDayLoad_shift(self.CHW_charging,self.shiftCount,self.totalCooling_load)
        return x


    #checks total hours the tank is on
    def total_on(self,tank_schedule,month):
        #loc is used to get the valu at the month and hour of the schedule using "labels- row or col name"
        value = tank_schedule.loc[tank_schedule["Month"]== month,"Total"]
        return value.iloc[0]


    @computed_field(return_type=MySeries)
    @property    
    def HL_totalShiftHours(self):

        result = []
        for ts in self.timeStamp['Date and Time']:
            month = ts.month

            result.append(self.total_on(self.hotTank_schedule, month))
        # print(df)
        return result
    






    @computed_field(return_type=MySeries)
    @property    
    def CL_totalShiftHours(self):

        result = []
        for ts in self.timeStamp['Date and Time']:
            month = ts.month

            result.append(self.total_on(self.coldTank_schedule, month))
        # print(df)
        return result
    


    def compute_CUP(self):
        df=pd.DataFrame({
            "Hot Load Shift Charging": self.HW_charging,
            "Cold Load Shift Charging": self.CHW_charging,
            "Hot Load Shift Hours":  self.HL_totalShiftHours,
            "Cold Load Shift Hours": self.CL_totalShiftHours,
            "Shift Count": self.shiftCount,
            "Predicted Day's Heating Load (Btu/h)": self.predictedDay_heating,
            "Predicted Day's Cooling Load (Btu/h)": self.predictedDay_cooling,
            "Predicted Heating Load in Load Shift Window (Btu/h)":self.predictedDay_heating_shift,
            "Predicted Cooling Load in Load Shift Window (Btu/h)":self.predictedDay_cooling_shift,

})
        
        return df


