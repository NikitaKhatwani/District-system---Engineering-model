from typing import Union
from pydantic import BaseModel, computed_field, validator
import pandas as pd
import numpy as np
from ucsbdistrict.CUP import CUP

class TES(BaseModel):
    # 
    # heatingLoad : pd.Series
    HW_upperTankVol : float 
    HW_lowerTankVol : float 
    HW_TankVol : float 
    HW_wallLoss : float 
    HW_ThermocloneLoss : float 
    HP_heatingCapacity: float 
    thermoclineDischarge: float 
    stopCharging: float 
    totalHeating_load : pd.Series 
    HW_districtSTP : pd.Series 
    districtHWRT : pd.Series 
    hotMaxflow: float
    coldMaxflow: float
    HW_upperTankTemp: float 
    HW_lowerTankTemp : float

    #CUP
    CUP_output_df : pd.DataFrame =None
    CUP_instance : CUP = None  # Default value None for Cup instance


    ##cold tank inputs
    CHW_upperTankVol : float
    CHW_lowerTankVol : float 
    CHW_TankVol : float 
    CHW_wallLoss : float 
    CHW_ThermocloneLoss : float 
    totalCooling_load: pd.Series
    districtCHWRT:pd.Series
    CHW_lowerTankTemp : float
    CHW_upperTankTemp : float

    #HP
    HP_possRecCHW : pd.Series = None
    HP_CHWST : pd.Series  = None

    #TES
    TES_hotCapacityHr : pd.Series = None


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
    def first_hotCapacityHr(self):

        # Calculate the value of interest (self.HW_upperTankVol / self.HW_TankVol)
        value = self.HW_upperTankVol / self.HW_TankVol

        # Repeat the calculated value 8760 times to create a Series
        series_values = np.repeat(value, 8760)

        # Create a pandas Series using the repeated values
        series = pd.Series(series_values)

        return series[0]
    
    
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
    def first_hotThafterLoss_supply(self):
        result = self.tempAfterLosses(self.first_hotCapacityHr,self.HW_upperTankVol,self.HW_wallLoss,self.HW_ThermocloneLoss)
        return result[0]

    def rest_hotThafterLoss_supply(self,index,hotCapacityHr,TES_H_th):
        result = self.tempAfterLosses(hotCapacityHr[index],TES_H_th[index-1],self.HW_wallLoss,self.HW_ThermocloneLoss)
        return result[0]
    

    
    @computed_field(return_type=MySeries)   
    @property
    def first_hotThafterLoss_return(self):
        result = self.tempAfterLosses(self.first_hotCapacityHr,self.HW_lowerTankVol,self.HW_wallLoss,self.HW_ThermocloneLoss)
        return result[0]
    

    def rest_hotThafterLoss_return(self,index,hotCapacityHr,TES_H_tc):
        result = self.tempAfterLosses(hotCapacityHr[index],TES_H_tc[index-1],self.HW_wallLoss,self.HW_ThermocloneLoss)
        return result[0]
    
    # @computed_field(return_type=MySeries)   
    # @property
    def hotThermocline(self,index,hotThafterLoss_supply,hotThafterLoss_return):
        result = self.HW_TankVol-hotThafterLoss_supply[index]-hotThafterLoss_return[index]
        return result









    # def compute(self):
    #     df = pd.DataFrame({
    #         "TES Hot Capacity hr-1 (%)": self.hotCapacityHr,
    #         "TES Hot Th after losses (Gal)_supply temp": self.hotThafterLoss_supply,
    #         "TES Hot Tc after losses (Gal)_return temp":self.hotThafterLoss_return,
    #         "TES Hot Thermocline (Gal)": self.hotThermocline,
    #         "TES Cold Th after losses (Gal)_supply temp": self.coldThafterLoss_supply,
    #         "TES Cold Tc after losses (Gal)_return temp":self.coldThafterLoss_return,
    #         "TES Cold Thermocline (Gal)": self.coldThermocline,

    #     },index = [0])
    #     return df
    

    # @computed_field(return_type=MySeries)   
    # @property
    # def CUP_output_df(self):
    #     result_series = self.CUP_instance.compute_CUP()
    #     print("result_series",result_series)
    #     return result_series




    # @computed_field(return_type=MySeries)   
    # @property
    def hotStatus(self,index,hotThafterLoss_supply,hotThermocline):
        # Apply conditions using pandas logical operations without direct if statements
        condition_hold = (self.CUP_output_df["Hot Load Shift Charging"][index] & ((hotThafterLoss_supply[index] + hotThermocline[index]) * self.thermoclineDischarge >= self.HW_TankVol * self.stopCharging) | (self.totalHeating_load[index] > self.HP_heatingCapacity))
        condition_charge = self.CUP_output_df["Hot Load Shift Charging"][index] & ~condition_hold
        condition_discharge = ~self.CUP_output_df["Hot Load Shift Charging"][index]

        # Assign corresponding values based on conditions
        value = np.select(
            [condition_hold, condition_charge, condition_discharge],
            ["Hold", "Charge", "Discharge"],
            default=None  # Default value if none of the conditions are met
        )

        return value

        # # Use numpy.select to assign results based on conditions``
        # result_series =   # Initialize result series with NA
        # result_series[condition_hold] = "Hold"
        # result_series[condition_charge] = "Charge"
        # result_series[~self.CUP_output_df["Hot Load Shift Charging"]] = "Discharge"
        
        # return result_series
        # # return self.CUP_output_df["Hot Load Shift Charging"]


    # @computed_field(return_type=MySeries)   
    # @property
    def hotLoadShiftChargeRate(self,index,hotStatus,hotThermocline,hotThafterLoss_return):
        mask_OG =  pd.Series((hotStatus[index] == "Discharge") | (hotStatus[index] == "Hold"))

        # mask_OG =  (hotStatus[index] == "Discharge") | (hotStatus[index] == "Hold")
        HW_STP = (self.HW_districtSTP[index]-self.districtHWRT[index])
        value1 = (self.HP_heatingCapacity-self.totalHeating_load[index])/500/HW_STP

        ##value2
        predictedHeatingMinusShift = self.CUP_output_df["Predicted Day's Heating Load (Btu/h)"][index]-self.CUP_output_df["Predicted Heating Load in Load Shift Window (Btu/h)"][index]
        mask = self.CUP_output_df["Hot Load Shift Charging"][index] !=0
        #initiate value with 0
        # value2 = 0
        # #apply condition when mask is TRue(not zero)
        # value2[mask] =  predictedHeatingMinusShift/500/HW_STP/self.CUP_output_df["Hot Load Shift Hours"]

        # Compute the value for value2 
        computed_value = predictedHeatingMinusShift/500/HW_STP/self.CUP_output_df["Hot Load Shift Hours"][index]
        if mask:
            value2 = computed_value.astype(int)  # Convert to int64 dtype
        else:
            value2 = 0


        ##value 3
        value3= hotThermocline[index]+hotThafterLoss_return[index]*self.thermoclineDischarge

        ##value 4
        value4 = self.hotMaxflow
        # print("v1",value1)
        # print("v2",value2)

        # print("v3",value3)
        # print(mask_OG)
        # print("cc",pd.Series([0]))
      
        result_series = pd.Series([0])
        # result_series = pd.Series([0]*8760)
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
        
        
        return result_series[0]




    def TES_H_flowInto(self,index,hotStatus,hotLoadShiftChargeRate):
        return np.where(hotStatus[index]== "Charge", hotLoadShiftChargeRate[index],0)

            

    def first_TES_H_temp(self):
        return self.HW_upperTankTemp
       
    

    def TES_H_flow(self,index,TES_H_temp):
        # print("xxx",self.totalHeating_load,"xxx",self.H_tempTES,"xxx",self.districtHWRT)
        return self.totalHeating_load[index]/500/(TES_H_temp[index]-self.districtHWRT[index]).astype(float)





    def TES_H_flowOut(self,index,TES_H_flow,hotThafterLoss_supply,hotStatus):

        minimum =min(TES_H_flow[index],self.hotMaxflow,(hotThafterLoss_supply[index]/60).astype(float))
        return np.where(hotStatus[index] =="Discharge",minimum,0).astype(float)
        # if self.hotStatus == "Discharge":
        #     return minimum  # Return the calculated minimum values
        # else:
        #     return 0  # Return 0 when not in "Discharge" state



    # @computed_field(return_type=MySeries)   
    # @property
    def TES_H_th(self,index,hotThafterLoss_supply,TES_H_flowInto,TES_H_flowOut):
        value1 = hotThafterLoss_supply[index]+TES_H_flowInto[index]*60
        value2 = np.where(TES_H_flowOut[index]*60>hotThafterLoss_supply[index],hotThafterLoss_supply[index],TES_H_flowOut[index]*60)
        return (value1 - value2)
    

    def TES_H_tc(self,index,hotThafterLoss_return,TES_H_flowInto,TES_H_flowOut):
        value1 = hotThafterLoss_return[index]+TES_H_flowOut[index]*60
        value2 = np.where(TES_H_flowInto[index]*60>hotThafterLoss_return[index],hotThafterLoss_return[index],TES_H_flowInto[index]*60)
        return (value1 - value2)


    def TES_H_capacity(self,index,TES_H_th):
        try:
            result = TES_H_th[index]/self.HW_TankVol
        except ValueError:
            result = 0
        return result


    def TES_H_Th_F(self,index,TES_H_flowInto,TES_H_temp,hotThafterLoss_supply):
        if TES_H_flowInto[index]==0:
            x = TES_H_temp[index]*hotThafterLoss_supply[index]+self.HW_districtSTP[index]*TES_H_temp[index]*60
            y = hotThafterLoss_supply[index]+TES_H_temp[index]*60
            return x/y
        else:
            return TES_H_temp[index]


    def TES_H_tc_F(self,index,TES_H_flowOut,TES_H_temp_tc,hotThafterLoss_return):
        if TES_H_flowOut[index] > 0:
            print(type(TES_H_flowOut),type(TES_H_temp_tc),type(hotThafterLoss_return),type(self.districtHWRT))
            return (TES_H_temp_tc[index]*hotThafterLoss_return[index]+self.districtHWRT[index]*TES_H_flowOut[index]*60)/(hotThafterLoss_return[index]+TES_H_flowOut[index]*60)
        else:
            return TES_H_temp_tc[index]



    def TES_H_calculate(self):
            hotCapacityHr=pd.Series(self.first_hotCapacityHr)
            hotThafterLoss_supply = pd.Series(self.first_hotThafterLoss_supply)
            hotThafterLoss_return = pd.Series(self.first_hotThafterLoss_return)
            hotThermocline = pd.Series(self.hotThermocline(0,hotThafterLoss_supply,hotThafterLoss_return))
            hotStatus = pd.Series(self.hotStatus(0,hotThafterLoss_supply,hotThermocline))
            hotLoadShiftChargeRate = pd.Series(self.hotLoadShiftChargeRate(0,hotStatus,hotThermocline,hotThafterLoss_return))
            TES_H_flowInto = pd.Series(self.TES_H_flowInto(0,hotStatus,hotLoadShiftChargeRate))
            TES_H_temp = pd.Series(self.first_TES_H_temp())
            TES_H_flow = pd.Series(self.TES_H_flow(0,TES_H_temp))
            TES_H_flowOut = pd.Series(self.TES_H_flowOut(0,TES_H_flow,hotThafterLoss_supply,hotStatus))
            TES_H_th = pd.Series(self.TES_H_th(0,hotThafterLoss_supply,TES_H_flowInto,TES_H_flowOut))
            TES_H_tc =pd.Series(self.TES_H_tc(0,hotThafterLoss_return,TES_H_flowInto,TES_H_flowOut))
            TES_H_capacity = pd.Series(self.TES_H_capacity(0,TES_H_th))
            TES_H_Th_F=pd.Series(self.TES_H_Th_F(0,TES_H_flowInto,TES_H_temp,hotThafterLoss_supply))
            TES_H_temp_tc = pd.Series(self.HW_lowerTankTemp)
            TES_H_tc_F = pd.Series(self.TES_H_tc_F(0,TES_H_flowOut,TES_H_temp_tc,hotThafterLoss_return))


            for index in range(1,5):
                #new formula from second value
                hotCapacityHr[index]=TES_H_capacity[index-1]
                hotThafterLoss_supply[index]=self.rest_hotThafterLoss_supply(index,hotCapacityHr,TES_H_th)
                hotThafterLoss_return[index] = self.rest_hotThafterLoss_return(index,hotCapacityHr,TES_H_tc)
                hotThermocline[index]= self.hotThermocline(index,hotThafterLoss_supply,hotThafterLoss_return)
                hotStatus[index] = self.hotStatus(index,hotThafterLoss_supply,hotThermocline)
                hotLoadShiftChargeRate[index] = self.hotLoadShiftChargeRate(index,hotStatus,hotThermocline,hotThafterLoss_return)
                TES_H_flowInto[index] = self.TES_H_flowInto(index,hotStatus,hotLoadShiftChargeRate)
                TES_H_temp[index] = TES_H_Th_F[index-1]
                TES_H_flow[index] = self.TES_H_flow(index,TES_H_temp)
                TES_H_flowOut[index] = self.TES_H_flowOut(index,TES_H_flow,hotThafterLoss_supply,hotStatus)
                TES_H_th[index] = self.TES_H_th(index,hotThafterLoss_supply,TES_H_flowInto,TES_H_flowOut)
                TES_H_tc[index] =self.TES_H_tc(index,hotThafterLoss_return,TES_H_flowInto,TES_H_flowOut)
                TES_H_capacity[index] = self.TES_H_capacity(index,TES_H_th)
                TES_H_Th_F[index]=self.TES_H_Th_F(index,TES_H_flowInto,TES_H_temp,hotThafterLoss_supply)
                TES_H_temp_tc[index] = TES_H_tc_F[index-1]
                TES_H_tc_F[index] = self.TES_H_tc_F(index,TES_H_flow,TES_H_temp_tc,hotThafterLoss_return)


            df3 = pd.DataFrame({
            "TES Hot Capacity hr-1 (%)": hotCapacityHr,
            "TES Hot Th after losses (Gal)_supply temp": hotThafterLoss_supply,
            "TES Hot Tc after losses (Gal)_return temp":hotThafterLoss_return,
            "TES Hot Thermocline (Gal)": hotThermocline,
            "TES Hot Status": hotStatus,
            "Hot Load-Shift Charge Rate (gpm)":hotLoadShiftChargeRate,
            "Flow into TES Hot Th (gpm)":TES_H_flowInto,
            "TES Hot Th Previous Hour (°F)":TES_H_temp ,
            "TES Flow to meet Campus Heating (gpm)":TES_H_flow,
            "Flow out of TES Hot Th (gpm)" : TES_H_flowOut,
            "TES Hot Th (°F)" : TES_H_Th_F,
            "TES Hot Th (Gal)" : TES_H_th,
            "TES Hot Tc (Gal)" : TES_H_tc,
            "TES Hot Capacity (%)" : TES_H_capacity,
            "districtHWRT":self.districtHWRT,
            "TES Hot Tc Previous Hour(°F)": TES_H_temp_tc,
            "TES Hot Tc (°F)": TES_H_tc_F
            
            
            # "TES Cold Th after losses (Gal)_supply temp": self.coldThafterLoss_supply,
            # "TES Cold Tc after losses (Gal)_return temp":self.coldThafterLoss_return,
            # "TES Cold Thermocline (Gal)": self.coldThermocline,
            },index = list(range(8760))
)     

            



            return df3








    # def compute_2(self):
    #     # return self.hotStatus
    #     df2 = pd.DataFrame({
    #         "TES Hot Status": self.hotStatus,
    #         "Hot Load-Shift Charge Rate (gpm)":self.hotLoadShiftChargeRate,
    #         "Flow into TES Hot Th (gpm)":self.H_flowIntoTES,
    #         "TES Hot Th Previous Hour (°F)":self.H_tempTES,
    #         "TES Flow to meet Campus Heating (gpm)":self.TES_H_flow,
    #         "Flow out of TES Hot Th (gpm)" : self.H_flowOutTES,
    #         "TES Hot Th (Gal)": self.TES_H_tc,
    #         "TES Hot Capacity (%)" : self.TES_H_capacity
    #     },index = [0])               
    #     return df2


################################## COLD TANK #####################################

    @computed_field(return_type=MySeries)   
    @property
    def first_coldCapacityHr(self):

        # Calculate the value of interest (self.HW_upperTankVol / self.HW_TankVol)
        value = self.CHW_lowerTankVol / self.CHW_TankVol

        # Repeat the calculated value 8760 times to create a Series
        series_values = np.repeat(value, 8760)

        # Create a pandas Series using the repeated values
        series = pd.Series(series_values)

        return series[0]
    
    @computed_field(return_type=MySeries)   
    @property
    def first_coldThafterLoss_supply(self):
        result = self.tempAfterLosses(self.first_coldCapacityHr,self.CHW_upperTankVol,self.CHW_wallLoss,self.CHW_ThermocloneLoss)
        return result[0]
    

    def rest_coldThafterLoss_supply(self,index,TES_C_th):
            result = self.tempAfterLosses(self.TES_hotCapacityHr[index],TES_C_th[index-1],self.CHW_wallLoss,self.CHW_ThermocloneLoss)
            return result[0]


    @computed_field(return_type=MySeries)   
    @property
    def first_coldThafterLoss_return(self):
        result = self.tempAfterLosses(self.first_coldCapacityHr,self.CHW_lowerTankVol,self.CHW_wallLoss,self.CHW_ThermocloneLoss)
        return result[0]
    
    def rest_coldThafterLoss_return(self,index,TES_C_tc):
        result = self.tempAfterLosses(self.TES_hotCapacityHr[index],TES_C_tc[index-1],self.CHW_wallLoss,self.CHW_ThermocloneLoss)
        return result[0]
    
    def coldThermocline(self,index,coldThafterLoss_supply,coldThafterLoss_return):
        # print("check TH",self.CHW_TankVol,coldThafterLoss_supply[index],coldThafterLoss_return[index] )
        result = self.CHW_TankVol-coldThafterLoss_supply[index]-coldThafterLoss_return[index]
        return result

    def first_TES_C_temp(self):
        return self.CHW_lowerTankTemp


    def TES_C_flow(self,index,TES_C_temp):
        # print("self.totalCooling_load",self.totalCooling_load)
        # print("ccdd0",self.districtCHWRT,"kk",self.HP_CHWST)
        return self.totalCooling_load[index]/500/(self.districtCHWRT[index]-TES_C_temp[index]).astype(float)

    @computed_field(return_type=MySeries)
    @property
    def coldCharge_window(self):
        
        # if self.CUP_output_df["Cold Load Shift Charging"]:
        #     if self.CUP_output_df["Predicted Day's Heating Load (Btu/h)"]>self.CUP_output_df["Predicted Day's Cooling Load (Btu/h)"]:
        #         result_series = True
        #     else:
        #         result_series = False
        # else:
        #     result_series = False
        value1 = self.CUP_output_df["Predicted Day's Heating Load (Btu/h)"]
        value2 = self.CUP_output_df["Predicted Day's Cooling Load (Btu/h)"]
        result_series = np.where(self.CUP_output_df["Cold Load Shift Charging"],np.where(value1>value2,False,True),False)
        result_series = pd.Series(result_series,index=self.CUP_output_df["Cold Load Shift Charging"].index)
        # print("coldchargewindow",result_series)
        return result_series

    def HP_C_flow(self,index):
        return self.totalCooling_load[index]/500/(self.districtCHWRT[index]-self.HP_CHWST[index]).astype(float)

    # def TES_H_flowInto(self,index,hotStatus,hotLoadShiftChargeRate):
    #     return np.where(hotStatus[index]== "Charge", hotLoadShiftChargeRate[index],0)



    def coldStatus(self,index,HP_C_flow,coldThafterLoss_return,coldThermocline,coldCharge_window):
        # define to conditions and then np.where is used to assign values across based on the index of the row
        condition1 = self.HP_possRecCHW[index] > HP_C_flow[index] or coldCharge_window[index]
        condition2 = coldThafterLoss_return[index] + coldThermocline[index] * self.thermoclineDischarge <= self.CHW_TankVol * self.stopCharging
 
        result = np.where(condition1,np.where(condition2,"Charge","Hold"),"Discharge")

        return result


    def coldLoadShiftChargeRate(self,index,coldStatus,coldCharge_window,coldThermocline,coldThafterLoss_supply):


        cond1 =  (coldStatus[index] == "Discharge") | (coldStatus[index] == "Hold")
        cond2 = coldCharge_window[index]
        value1 =self.CUP_output_df["Predicted Day's Cooling Load (Btu/h)"][index]-self.CUP_output_df["Predicted Cooling Load in Load Shift Window (Btu/h)"][index]
        value2 = value1/500/(self.districtCHWRT[index]-self.HP_CHWST[index])/self.CUP_output_df["Cold Load Shift Hours"][index]
        
        max_value = np.where(self.CUP_output_df["Cold Load Shift Hours"][index]==0,0,value2)
        charge_value = (coldThafterLoss_supply[index]+ coldThermocline[index]*self.thermoclineDischarge)/60
        min_value = np.minimum(self.totalCooling_load[index]/500/(self.districtCHWRT[index]-self.HP_CHWST[index]),charge_value)
            # Check type and shape of key variables
        # print(f"Type of charge_value: {type(charge_value)}, Shape: {charge_value.shape},",charge_value)
        # print(f"Type of min_value: {type(min_value)}, Shape: {min_value.shape}",min_value)
        # print(f"check yhid too : {type(self.HP_possRecCHW[index]),self.HP_possRecCHW[index]} and {type(self.coldMaxflow),self.coldMaxflow}")
        result = np.where(
                cond1,
                0,
                np.where(
                    cond2,
                    np.minimum(np.maximum(max_value, self.HP_possRecCHW[index]), charge_value, np.array(self.coldMaxflow)),
                    np.minimum(self.HP_possRecCHW[index] - min_value, self.coldMaxflow)
                )
            )
        # print("cond", cond1, cond2)
        # print(result)
        return result


    
    def TES_C_flowInto(self,index,coldStatus,coldLoadShiftChargeRate):
        return np.where(coldStatus[index]== "Charge", coldLoadShiftChargeRate[index],0)
   
    

    # def TES_H_flow(self,index,TES_H_temp):
    #     # print("xxx",self.totalHeating_load,"xxx",self.H_tempTES,"xxx",self.districtHWRT)
    #     return self.totalHeating_load[index]/500/(TES_H_temp[index]-self.districtHWRT[index]).astype(float)



    def TES_C_flowOut(self,index,TES_C_flow,coldThafterLoss_return,coldStatus):

        minimum =min(TES_C_flow[index],self.coldMaxflow,(coldThafterLoss_return[index]/60).astype(float))
        return np.where(coldStatus[index] =="Discharge",minimum,0).astype(float)


    def TES_C_th(self,index,coldThafterLoss_supply,TES_C_flowInto,TES_C_flowOut):
        # print(TES_C_flowInto[index],coldThafterLoss_supply[index],coldThafterLoss_supply[index],TES_C_flowInto[index]),TES_C_flowOut)
        value1 = np.where(TES_C_flowInto[index]*60>coldThafterLoss_supply[index],coldThafterLoss_supply[index],TES_C_flowInto[index]*60)+TES_C_flowOut[index]*60
        return (coldThafterLoss_supply[index] - value1)
    

    def TES_C_tc(self,index,coldThafterLoss_return,TES_C_flowInto,TES_C_flowOut):
        # print(type(TES_C_flowInto))
        value1 = coldThafterLoss_return[index]+TES_C_flowInto[index]*60
        value2 = np.where(TES_C_flowOut[index]*60>coldThafterLoss_return[index],coldThafterLoss_return[index],TES_C_flowOut[index]*60)
        return (value1 - value2)


    # def TES_H_capacity(self,index,TES_H_th):
    #     try:
    #         result = TES_H_th[index]/self.HW_TankVol
    #     except ValueError:
    #         result = 0
    #     return result


    # def TES_H_Th_F(self,index,TES_H_flowInto,TES_H_temp,hotThafterLoss_supply):
    #     if TES_H_flowInto[index]==0:
    #         x = TES_H_temp[index]*hotThafterLoss_supply[index]+self.HW_districtSTP[index]*TES_H_temp[index]*60
    #         y = hotThafterLoss_supply[index]+TES_H_temp[index]*60
    #         return x/y
    #     else:
    #         return TES_H_temp[index]


    def TES_C_tc_F(self,index,TES_C_flowInto,TES_C_temp,coldThafterLoss_return):
        if TES_C_flowInto[index] > 0:

            return (TES_C_temp[index]*coldThafterLoss_return[index]+(self.HP_CHWST[index]*TES_C_flowInto[index]*60))/(coldThafterLoss_return[index]+TES_C_flowInto[index]*60)
        else:
            return TES_C_temp[index]
        

    def TES_C_capacity(self,index,TES_C_tc_F,TES_C_tc):
        print("trial",TES_C_tc_F[index],self.CHW_lowerTankTemp,TES_C_tc[index],self.CHW_TankVol)
        try:
            epsilon = 1e-6  # Adjust epsilon based on your tolerance requirement
            if abs(TES_C_tc_F[index] - self.CHW_lowerTankTemp) <= epsilon:
            # if TES_C_tc_F[index] <= self.CHW_lowerTankTemp:
                result = TES_C_tc[index]/self.CHW_TankVol
            else:
                result = 0
        except ValueError:
            result = 0
        return result

    # def TES_H_capacity(self,index,TES_H_th):
    #     try:
    #         result = TES_H_th[index]/self.HW_TankVol
    #     except ValueError:
    #         result = 0
    #     return result




    def TES_C_calculate(self):

                coldCapacityHr=pd.Series(self.first_coldCapacityHr)
                coldThafterLoss_supply = pd.Series(self.first_coldThafterLoss_supply)
                coldThafterLoss_return = pd.Series(self.first_coldThafterLoss_return)
                coldThermocline = pd.Series(self.coldThermocline(0,coldThafterLoss_supply,coldThafterLoss_return))
                TES_C_temp = pd.Series(self.first_TES_C_temp())
                TES_C_flow = pd.Series(self.TES_C_flow(0,TES_C_temp))
                HP_C_flow = pd.Series(self.HP_C_flow(0))
                coldCharge_window = self.coldCharge_window

                coldStatus = pd.Series(self.coldStatus(0,HP_C_flow,coldThafterLoss_return,coldThermocline,coldCharge_window))
                coldLoadShiftChargeRate=pd.Series(self.coldLoadShiftChargeRate(0,coldStatus,coldCharge_window,coldThermocline,coldThafterLoss_supply))
                
                TES_C_flowInto= pd.Series(self.TES_C_flowInto(0,coldStatus,coldLoadShiftChargeRate))
                TES_C_flowOut = pd.Series(self.TES_C_flowOut(0,TES_C_flow,coldThafterLoss_return,coldStatus))

                TES_C_th = pd.Series(self.TES_C_th(0,coldThafterLoss_supply,TES_C_flowInto,TES_C_flowOut))
                TES_C_tc =pd.Series(self.TES_C_tc(0,coldThafterLoss_return,TES_C_flowInto,TES_C_flowOut))
                TES_C_tc_F = pd.Series(self.TES_C_tc_F(0,TES_C_flowInto,TES_C_temp,coldThafterLoss_return))
                TES_C_capacity = pd.Series(self.TES_C_capacity(0,TES_C_tc_F,TES_C_tc))

                for index in range(1,5):

                    #new formula from second value
                    coldCapacityHr[index]=TES_C_capacity[index-1]
                    coldThafterLoss_supply[index]=self.rest_coldThafterLoss_supply(index,TES_C_th)
                    coldThafterLoss_return[index] = self.rest_coldThafterLoss_return(index,TES_C_tc)
                    coldThermocline[index]= self.coldThermocline(index,coldThafterLoss_supply,coldThafterLoss_return)
                    TES_C_temp[index] = self.first_TES_C_temp()
                    TES_C_flow[index] = self.TES_C_flow(index,TES_C_temp)
                    HP_C_flow[index] = self.HP_C_flow(index)
                    coldCharge_window = self.coldCharge_window

                    coldStatus[index] = self.coldStatus(index,HP_C_flow,coldThafterLoss_return,coldThermocline,coldCharge_window)

                    coldLoadShiftChargeRate[index]=self.coldLoadShiftChargeRate(index,coldStatus,coldCharge_window,coldThermocline,coldThafterLoss_supply)
                    
                    TES_C_flowInto[index]= self.TES_C_flowInto(index,coldStatus,coldLoadShiftChargeRate)
                    TES_C_flowOut[index] = self.TES_C_flowOut(index,TES_C_flow,coldThafterLoss_return,coldStatus)
                    TES_C_th[index] = self.TES_C_th(index,coldThafterLoss_supply,TES_C_flowInto,TES_C_flowOut)
                    TES_C_tc[index] = self.TES_C_tc(index,coldThafterLoss_return,TES_C_flowInto,TES_C_flowOut)
                    TES_C_tc_F[index] = self.TES_C_tc_F(index,TES_C_flowInto,TES_C_temp,coldThafterLoss_return)
                    TES_C_capacity[index] = self.TES_C_capacity(index,TES_C_tc_F,TES_C_tc)


                    # hotLoadShiftChargeRate[index] = self.hotLoadShiftChargeRate(index,hotStatus,hotThermocline,hotThafterLoss_return)
                    # TES_H_flowInto[index] = self.TES_H_flowInto(index,hotStatus,hotLoadShiftChargeRate)
                    # TES_H_temp[index] = TES_H_Th_F[index-1]
                    # TES_H_flow[index] = self.TES_H_flow(index,TES_H_temp)
                    # TES_H_flowOut[index] = self.TES_H_flowOut(index,TES_H_flow,hotThafterLoss_supply,hotStatus)
                    # TES_H_th[index] = self.TES_H_th(index,hotThafterLoss_supply,TES_H_flowInto,TES_H_flowOut)
                    # TES_H_tc[index] =self.TES_H_tc(index,hotThafterLoss_return,TES_H_flowInto,TES_H_flowOut)
                    # TES_H_capacity[index] = self.TES_H_capacity(index,TES_H_th)
                    # TES_H_Th_F[index]=self.TES_H_Th_F(index,TES_H_flowInto,TES_H_temp,hotThafterLoss_supply)
                    


                df4 = pd.DataFrame({
                "TES Cold Capacity hr-1 (%)": coldCapacityHr,
                "TES Cold Th after losses (Gal)_supply temp": coldThafterLoss_supply,
                "TES Cold Tc after losses (Gal)_return temp":coldThafterLoss_return,
                "TES Cold Thermocline (Gal)": coldThermocline,
                "TES Cold Charge Window Adjustment":coldCharge_window,
                "TES Cold Status": coldStatus,
                "Cold Load-Shift Charge Rate (gpm)":coldLoadShiftChargeRate,
                # "Flow into TES Hot Th (gpm)":TES_H_flowInto,
                "TES Cold Tc Previous Hour (°F)":TES_C_temp ,
                "TES Flow to meet Campus Heating (gpm)":TES_C_flow,
                # "Flow out of TES Hot Th (gpm)" : TES_H_flowOut,
                # "TES Hot Th (°F)":TES_H_Th_F,
                # "TES Hot Capacity (%)" : TES_H_capacity,
                # "districtHWRT":self.districtHWRT
                "HP Flow to meet Campus Cooling (gpm)": HP_C_flow,
                "Flow into TES Cold Tc (gpm)": TES_C_flowInto,
                "Flow out of TES Cold Tc (gpm)":TES_C_flowOut,
                "TES Cold Th (Gal)": TES_C_th,
                "TES Cold Tc (Gal)": TES_C_tc,
                "TES Cold Tc (°F)": TES_C_tc_F,
                "TES Cold Capacity (%)" : TES_C_capacity,
                # "TES Cold Th after losses (Gal)_supply temp": self.coldThafterLoss_supply,
                # "TES Cold Tc after losses (Gal)_return temp":self.coldThafterLoss_return,
                # "TES Cold Thermocline (Gal)": self.coldThermocline,
                },index = list(range(8760))
    )     

                



                return df4