from pydantic import BaseModel

class BuildingParameters(BaseModel):
    supply_temp_degF: float = 2
    ...

class Building(BaseModel):
    parameters: BuildingParameters
    caan_nb: int
    name: str
    cooling_load: pd.Series
        
    def some_calculation(self):
        return self.parameters.supply_temp_degF * 10
    
    params_bldg_1 = BuildingParameters(supply_temp_degF=10)
bldg_1 = Building(parameters=params_bldg_1, caan_nb=1000, name="Lab", cooling_load=pd.Series())

bldg_1.some_calculation()

module_1.calculate_blah()
print(module_1.DHWRt)

from pydantic import BaseModel, computed_field

#Building module class
class buildingModule(BaseModel):
#     parameters: #
    LoopSTP : pd.Series = None # Assuming it is a pandas Series (column from a DataFrame)
    supplyLosses : int = None
    ST : pd.Series = None
    deltaT_Max : int = None
    deltaT_Min : int = None
    maxLoad : float = None
    coolingLoad : pd.Series = None
    RT : pd.Series = None
    data : pd.Series = None
    target : pd.Series = None
    ind : pd.Series = None
    delta : pd.Series = None
    load : pd.Series = None 
    DHWtemp : pd.Series = None 
    minLoadApproach : int = None
    DHflow : pd.Series = None 
    HHflow : pd.Series = None 
    HHstp  : pd.Series = None 
    HWSflow : pd.Series = None 
    bypassHHWS : pd.Series = None 
    
    @property
    def DHWRt(self) -> pd.Series:
        return pd.Series()  # Add calculation here
    
    @property
    def CHWRt(self) -> pd.Series
        return pd.Series()
    
    def compute(self):
        return pd.DataFrame([self.DHWRt, self.CHWRt, ])
    
    class Config:
        arbitrary_types_allowed = True
    
    def calculate_sum(self) -> pd.Series:
        return self.LoopSTP + self.supplyLosses
    
    def calculate_CHWRT(self) -> pd.Series:
        return self.ST+((self.deltaT_Max-self.deltaT_Min)/self.maxLoad)*self.coolingLoad+self.deltaT_Min
    
    def calculate_flow(self) -> pd.Series:
            print("self.load ",self.load )
            return  self.load / 500 / abs(self.RT - self.ST)
        
    
    def find_min_index(self):
        # Convert data_series and target_series to NumPy arrays for numerical operations
        data_values = self.data.values
#         print("hi",self.HHWstp)
        target_values = self.target.values
        
        print(data_values.shape,data_values)
        # Reshape self.data to (1, 4) to broadcast across self.target
        data_values_reshaped = data_values.reshape(1, 4)  # Shape (1, 4)

        # Subtract each element of data_values_reshaped from every element of self.target
        diff = target_values[:, np.newaxis] - data_values_reshaped  # Broadcasting happens here
        # diff now has shape (8760, 4), where each row contains the differences

        # Find the index of the minimum absolute difference for each element in self.target
        min_index = np.argmin(np.abs(diff), axis=1)
        # min_index now has shape (8760,) containing indices of minimum absolute differences
        
        return min_index

    def match_index(self):
          try:
            # Create a new Series with sequential index and values from self.delta
            new_deltaSeries = pd.Series(self.delta[self.ind].values, index=range(len(self.ind)))
            print(self.delta[self.ind])
            return self.target-new_deltaSeries
          except IndexError:
            return None
        
    def match_DHWst(self):
        return self.ST[self.ind]


    
    def calculate_DHWrt(self):
        self.DHWrt = self.DHWtemp + np.maximum(self.deltaT_Min, (self.deltaT_Max-self.deltaT_Min)/(1-self.minLoadApproach/self.maxLoad)*self.load/self.maxLoad+\
                                                     self.deltaT_Max-(self.deltaT_Max-self.deltaT_Min)/(1-self.minLoadApproach/self.maxLoad))
        return self.DHWrt
        
    def calculate_HWSflow(self):
        return self.DHflow +(self.HHflow*(self.HHstp-self.RT))/(self.ST-self.RT)
    
    def calculate_bypassHHWS(self):
        return self.HHflow-(self.HWSflow-self.DHflow)
    
    def calculate_HWRflow(self):
        return self.HHflow-self.bypassHHWS+self.DHflow
    
    def calculate_HWRT(self):
        return (self.RT*(self.HHflow-self.bypassHHWS)+self.DHflow*self.DHWrt)/self.HWSflow
    
    
# HWRT = buildingModule(RT=buildingModule_outputs[Building HHWRT (°F)],DHWrt = buildingModule_outputs["Building DHWRT (°F)"],\
#                       bypassHHWS = buildingModule_outputs["Bypassed Return to HHWS (gpm)"],DHflow = buildingModule_outputs["Building DHWR Flow (gpm)"],\
#                       HHflow =buildingModule_outputs["Building HHWR Flow (gpm)"],HWSflow = buildingModule_outputs["District HWS Flow (gpm)"]) 


