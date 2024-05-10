from typing import Union
from pydantic import BaseModel, computed_field, validator
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

from ucsbdistrict.inputs import dateTime,weather_df



#########################  import Building COP and process load Data #######################################
buildingLoadData_df = pd.read_excel('UCSB merged baselines_COP_process load.xlsx', sheet_name = "Analytics")
# Strip whitespace from all elements in the column
buildingLoadData_df['Simulation_id'] = buildingLoadData_df['Simulation_id'].str.strip()





####################### Process Load COP_df defined ###############
COP_df = pd.DataFrame()
COP_df["Names"] = ["COOLCOP","HEATCOP","KITCHCOP","DHWCOP","LAUNCOP","Other"]
COP_df["Electric"] = ["Default",(1,4),0.8,1,1,1]
COP_df["Gas"] = ["-",(0,0.99),0.5,0.6,1,1]
COP_df["CHW"] = [-1,"-","-","-","-",1]
COP_df["HW/Steam"] = ["-",-1,"-",-1,"-",1]
COP_df.set_index("Names", inplace=True)


##########################  upload PNNL profiles fo all utilites #######################################

# PNNL electric profiles
pnnl_hotWater_e_profiles_df = pd.read_excel('Building_PNNL_elec_profiles.xlsx', sheet_name = "Hot Water Load (kbtu)")
pnnl_cooking_e_profiles_df = pd.read_excel('Building_PNNL_elec_profiles.xlsx', sheet_name = "Cooking Load (kbtu)")
pnnl_laundry_e_profiles_df = pd.read_excel('Building_PNNL_elec_profiles.xlsx', sheet_name = "Laundry Load (kbtu)")
pnnl_other_e_profiles_df = pd.read_excel('Building_PNNL_elec_profiles.xlsx', sheet_name = "Other Process Load (kbtu)")

#PNNL gas profiles
pnnl_hotWater_g_profiles_df = pd.read_excel('Building_PNNL_gas_profiles.xlsx', sheet_name = "Hot Water Load (kbtu)")
pnnl_cooking_g_profiles_df = pd.read_excel('Building_PNNL_gas_profiles.xlsx', sheet_name = "Cooking Load (kbtu)")
pnnl_laundry_g_profiles_df = pd.read_excel('Building_PNNL_gas_profiles.xlsx', sheet_name = "Laundry Load (kbtu)")
pnnl_other_g_profiles_df = pd.read_excel('Building_PNNL_gas_profiles.xlsx', sheet_name = "Other Process Load (kbtu)")

#PNNL gas profiles
pnnl_hotWater_c_profiles_df = pd.read_excel('Building_PNNL_c_profiles.xlsx', sheet_name = "Hot Water Load (kbtu)")
pnnl_cooking_c_profiles_df = pd.read_excel('Building_PNNL_c_profiles.xlsx', sheet_name = "Cooking Load (kbtu)")
pnnl_laundry_c_profiles_df = pd.read_excel('Building_PNNL_c_profiles.xlsx', sheet_name = "Laundry Load (kbtu)")
pnnl_other_c_profiles_df = pd.read_excel('Building_PNNL_c_profiles.xlsx', sheet_name = "Other Process Load (kbtu)")

#PNNL gas profiles
pnnl_hotWater_s_profiles_df = pd.read_excel('Building_PNNL_s_profiles.xlsx', sheet_name = "Hot Water Load (kbtu)")
pnnl_cooking_s_profiles_df = pd.read_excel('Building_PNNL_s_profiles.xlsx', sheet_name = "Cooking Load (kbtu)")
pnnl_laundry_s_profiles_df = pd.read_excel('Building_PNNL_s_profiles.xlsx', sheet_name = "Laundry Load (kbtu)")
pnnl_other_s_profiles_df = pd.read_excel('Building_PNNL_s_profiles.xlsx', sheet_name = "Other Process Load (kbtu)")



######################################### calculate loads #######################################################

building_usage_folder =  r"C:\Users\nikita.khatwani\Documents\UCSB\District sytem - Engineering model\District system - Engineering model\Building loads data"
# Get a list of all files in the folder
files = os.listdir(building_usage_folder)
allBldgLoads_output =pd.DataFrame()
allBldgElecUse_output = pd.DataFrame()
allBldgGasUse_output = pd.DataFrame()

#loop through all files
for file in files:

    file_path = os.path.join(building_usage_folder, file)
    buildingLoads_perSqFt_df = pd.read_csv(file_path)
    id = file.replace(".csv", "").strip()
    # Remove the "in_" prefix using str.replace()
    id= id.replace("in_", "")


    caan_no = int(buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"CAAN"].iloc[0])
    print("caan_no",caan_no) 
    area = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"Area [sf]"].iloc[0] 
    coolCOP = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"COOLCOP"].iloc[0] 
    heatCOP = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"HEATCOP"].iloc[0] 


    # annual process usage
    e_annual_processUsage = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"E_process (kBtu/sf)"].iloc[0]
    g_annual_processUsage = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"G_process"].iloc[0]
    c_annual_processUsage = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"C_process"].iloc[0]
    s_annual_processUsage = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"S_process"].iloc[0]


    program = buildingLoadData_df.loc[buildingLoadData_df["Simulation_id"]==id,"Program"].iloc[0]

    ################ PNNL profiles
    #electric profiles
    hotWater_e_programProfiles = pnnl_hotWater_e_profiles_df[program]
    cooking_e_programProfiles = pnnl_cooking_e_profiles_df[program]
    laundry_e_programProfiles = pnnl_laundry_e_profiles_df[program]
    other_e_programProfiles = pnnl_other_e_profiles_df[program]

    #gas profiles
    hotWater_g_programProfiles = pnnl_hotWater_g_profiles_df[program]
    cooking_g_programProfiles = pnnl_cooking_g_profiles_df[program]
    laundry_g_programProfiles = pnnl_laundry_g_profiles_df[program]
    other_g_programProfiles = pnnl_other_g_profiles_df[program]

    #c profiles
    hotWater_c_programProfiles = pnnl_hotWater_c_profiles_df[program]
    cooking_c_programProfiles = pnnl_cooking_c_profiles_df[program]
    laundry_c_programProfiles = pnnl_laundry_c_profiles_df[program]
    other_c_programProfiles = pnnl_other_c_profiles_df[program]

    #s profiles
    hotWater_s_programProfiles = pnnl_hotWater_s_profiles_df[program]
    cooking_s_programProfiles = pnnl_cooking_s_profiles_df[program]
    laundry_s_programProfiles = pnnl_laundry_s_profiles_df[program]
    other_s_programProfiles = pnnl_other_s_profiles_df[program]
    
    #################### load calculation ##########

    ###### empty df ################
    buildingModule_output = pd.DataFrame()
    simulationThermalLoads_output = pd.DataFrame()
    simulationElecUse_output = pd.DataFrame()
    simulationGasUse_output = pd.DataFrame()


    #define dateTime and CAAN no.
    simulationThermalLoads_output["Timestamp"] = dateTime
    simulationThermalLoads_output["Building ID CAAN"] = [caan_no] * 8760

    simulationElecUse_output["Timestamp"] = dateTime
    simulationElecUse_output["Building ID CAAN"] = [caan_no] * 8760

    simulationGasUse_output["Timestamp"] = dateTime
    simulationGasUse_output["Building ID CAAN"] = [caan_no] * 8760

    # cooling and heating loads
    simulationThermalLoads_output["Cooling Load (kbtu)"] = buildingLoads_perSqFt_df["cooling.load.kBtu_per_sqft"]*area
    simulationThermalLoads_output["Heating Load (kbtu)"] = buildingLoads_perSqFt_df["heating.load.kBtu_per_sqft"]*area
    

    #heating anc cooling usage
    simulationElecUse_output["Cooling (kWh)"]= simulationThermalLoads_output["Cooling Load (kbtu)"]/coolCOP
    if heatCOP in range(1,5):
        simulationElecUse_output["Heating (kWh)"]= simulationThermalLoads_output["Heating Load (kbtu)"]/heatCOP
        simulationGasUse_output["Heating (Therms)"]= [0]*8760
    elif 0 < heatCOP < 1:
        print("caan",caan_no,heatCOP)
        simulationElecUse_output["Heating (kWh)"]= [0]*8760
        simulationGasUse_output["Heating (Therms)"]= simulationThermalLoads_output["Heating Load (kbtu)"]/heatCOP
    else: #incase of district heating 
        print("caan2",caan_no,heatCOP)
        simulationElecUse_output["Heating (kWh)"]= [0]*8760
        simulationGasUse_output["Heating (Therms)"]= [0]*8760 

    # electric process usage
    simulationElecUse_output["Hot Water (kWh)"] = e_annual_processUsage*area*hotWater_e_programProfiles
    simulationElecUse_output["Cooking (kWh)"] = e_annual_processUsage*area*cooking_e_programProfiles
    simulationElecUse_output["Laundry (kWh)"] = e_annual_processUsage*area*laundry_e_programProfiles
    simulationElecUse_output["Other Process (kWh)"] = e_annual_processUsage*area*other_e_programProfiles


    # electric process loads
    buildingModule_output["e_Hot Water Load (kbtu)"] = simulationElecUse_output["Hot Water (kWh)"]*COP_df.loc["DHWCOP","Electric"]
    buildingModule_output["e_Cooking Load (kbtu)"] = simulationElecUse_output["Cooking (kWh)"]*COP_df.loc["KITCHCOP","Electric"]
    buildingModule_output["e_Laundry Load (kbtu)"] = simulationElecUse_output["Laundry (kWh)"]*COP_df.loc["LAUNCOP","Electric"]
    # buildingModule_output["e_Other Process Load (kbtu)"] = simulationElecUse_output["Other Process Load (kWh)"]*COP_df.loc["Other","Electric"]


    # gas process usage
    simulationGasUse_output["Hot Water (Therms)"] = g_annual_processUsage*area*hotWater_g_programProfiles
    simulationGasUse_output["Cooking (Therms)"] = g_annual_processUsage*area*cooking_g_programProfiles
    simulationGasUse_output["Laundry (Therms)"] = g_annual_processUsage*area*laundry_g_programProfiles
    simulationGasUse_output["Other Process (Therms)"] = g_annual_processUsage*area*other_g_programProfiles


    # gas process loads
    buildingModule_output["g_Hot Water Load (kbtu)"] = simulationGasUse_output["Hot Water (Therms)"]*COP_df.loc["DHWCOP","Gas"]
    buildingModule_output["g_Cooking Load (kbtu)"] = simulationGasUse_output["Cooking (Therms)"] *COP_df.loc["KITCHCOP","Gas"]
    buildingModule_output["g_Laundry Load (kbtu)"] = simulationGasUse_output["Laundry (Therms)"]*COP_df.loc["LAUNCOP","Gas"]
    # buildingModule_output["g_Other Process Load (kbtu)"] = simulationGasUse_output["Other Process Load (Therms)"]*COP_df.loc["Other","Gas"]

    # CHW process loads
    buildingModule_output["c_Other Process Load (kbtu)"] = c_annual_processUsage*area*other_c_programProfiles*COP_df.loc["Other","CHW"]

    # HW/Steam process loads
    buildingModule_output["s_Hot Water Load (kbtu)"] = s_annual_processUsage*area*hotWater_s_programProfiles*COP_df.loc["DHWCOP","HW/Steam"]
    # buildingModule_output["s_Other Process Load (kbtu)"] = s_annual_processUsage*area*other_s_programProfiles*COP_df.loc["Other","HW/Steam"]



    # Total process loads
    simulationThermalLoads_output["Hot Water Load (kbtu)"] = buildingModule_output["e_Hot Water Load (kbtu)"] + buildingModule_output["g_Hot Water Load (kbtu)"] + buildingModule_output["s_Hot Water Load (kbtu)"]
    simulationThermalLoads_output["Cooking Load (kbtu)"] = buildingModule_output["e_Cooking Load (kbtu)"]+buildingModule_output["g_Cooking Load (kbtu)"]
    simulationThermalLoads_output["Laundry Load (kbtu)"] = buildingModule_output["e_Laundry Load (kbtu)"] + buildingModule_output["g_Laundry Load (kbtu)"]
    # simulationThermalLoads_output["Other Process Load (kbtu)"] = buildingModule_output["e_Other Process Load (kbtu)"]+buildingModule_output["g_Other Process Load (kbtu)"]+buildingModule_output["c_Other Process Load (kbtu)"]+buildingModule_output["s_Other Process Load (kbtu)"]


    ############### Loading other elec process load/usage(load=usage for these) directly from CS results ############
    simulationElecUse_output["Plug Loads (kWh)"]= buildingLoads_perSqFt_df["equipment.elec.kBtu_per_sqft"]*area
    simulationElecUse_output["Lighting (kWh)"]= buildingLoads_perSqFt_df["lighting.elec.kBtu_per_sqft"]*area
    simulationElecUse_output["Fans (kWh)"]= buildingLoads_perSqFt_df["fans.elec.kBtu_per_sqft"]*area
    simulationElecUse_output["Pumps (kWh)"]= buildingLoads_perSqFt_df["pumps.elec.kBtu_per_sqft"]*area
    # simulationElecUse_output["Misc. (kWh)"]= buildingLoads_perSqFt_df["misc.elec.kBtu_per_sqft"]*area


    allBldgLoads_output = pd.concat([allBldgLoads_output, simulationThermalLoads_output], ignore_index=True)  # Concatenate new row to existing DataFra\\\\\]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]'
    allBldgElecUse_output = pd.concat([allBldgElecUse_output, simulationElecUse_output], ignore_index=True)  # Concatenate new row to existing DataFra\\\\\]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]'
    allBldgGasUse_output = pd.concat([allBldgGasUse_output, simulationGasUse_output], ignore_index=True)  # Concatenate new row to existing DataFra\\\\\]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]'



    ############# Current District Therm Op loads #############
    building_meta_df = pd.read_excel("Building_MetaData_.xlsx",header=1)
    current_district_therm_loads = pd.DataFrame()
    year = 2045


    # Define a function to check conditions for each building
    def meets_district_cooling_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year < row["Year of Decarb"] and
            row["Current District Cooling Y/N?"] == "Y"
        )
        return condition
    
    # Define a function to check conditions for each building
    def meets_district_heating_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year < row["Year of Decarb"] and
            row["Current District Heating Y/N?"] == "Y"
        )
        return condition
    
        # Define a function to check conditions for each building
    def meets_district_DHW_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year < row["Year of Decarb"] and
            row["Current District Hot Water Y/N?"] == "Y"
        )
        return condition
    
    def filtered_bldg(building_meta_df,allBldg_output,meets_conditions1,meets_conditions2,meets_conditions3,meets_conditions4,df,type,usage,units):
        # cooling filter 
        # Apply the condition check to filter buildings
        filtered_buildings_cooling = building_meta_df.apply(meets_conditions1, axis=1)

        # Extract the Building ID CAAN values of buildings that meet the conditions
        filtered_building_ids_cooling = building_meta_df.loc[filtered_buildings_cooling, "Building ID CAAN"].tolist()

        # Filter cooling loads for buildings that meet the conditions
        filtered_loads_usage_cooling = allBldg_output[allBldg_output["Building ID CAAN"].isin(filtered_building_ids_cooling)]
        print("filtered_building_ids_cooling ",filtered_buildings_cooling)
        print("filtered_loads_usage_cooling ",filtered_loads_usage_cooling)
        # heating filter 
        filtered_buildings_heating = building_meta_df.apply(meets_conditions2, axis=1)

        # Extract the Building ID CAAN values of buildings that meet the conditions
        filtered_building_ids_heating = building_meta_df.loc[filtered_buildings_heating, "Building ID CAAN"].tolist()

        # Filter cooling loads for buildings that meet the conditions
        filtered_loads_usage_heating = allBldg_output[allBldg_output["Building ID CAAN"].isin(filtered_building_ids_heating)]

        # DHW filter
        filtered_buildings_DHW = building_meta_df.apply(meets_conditions3, axis=1)

        # Extract the Building ID CAAN values of buildings that meet the conditions
        filtered_building_ids_DHW = building_meta_df.loc[filtered_buildings_DHW, "Building ID CAAN"].tolist()

        # Filter cooling loads for buildings that meet the conditions
        filtered_loads_usage_DHW = allBldg_output[allBldg_output["Building ID CAAN"].isin(filtered_building_ids_DHW)]

        # Other filter
        filtered_buildings_other = building_meta_df.apply(meets_conditions4, axis=1)

        # Extract the Building ID CAAN values of buildings that meet the conditions
        filtered_building_ids_other = building_meta_df.loc[filtered_buildings_other, "Building ID CAAN"].tolist()

        # Filter cooling loads for buildings that meet the conditions
        filtered_loads_usage_other = allBldg_output[allBldg_output["Building ID CAAN"].isin(filtered_building_ids_other)]

        # Sum the cooling loads across all hours for these buildings
        if usage == "Load (kBtu)":
            df[f"Current {type} Cooling {usage}"] = filtered_loads_usage_cooling.groupby(allBldg_output['Timestamp'])["Cooling Load (kbtu)"].sum()
            df[f"Current {type} Heating {usage}"] = filtered_loads_usage_heating.groupby(allBldg_output['Timestamp'])["Heating Load (kbtu)"].sum()
            df[f"Current {type} Hot Water {usage}"] = filtered_loads_usage_DHW.groupby(allBldg_output['Timestamp'])["Hot Water Load (kbtu)"].sum()
            if type != "District":
                df[f"Current {type} Cooking {usage}"] = filtered_loads_usage_other.groupby(allBldg_output['Timestamp'])["Cooking Load (kbtu)"].sum()
                df[f"Current {type} Laundry {usage}"] = filtered_loads_usage_other.groupby(allBldg_output['Timestamp'])["Laundry Load (kbtu)"].sum()
        else:
            df[f"Current {type} Heating {usage}"] = filtered_loads_usage_heating.groupby(allBldg_output['Timestamp'])[f"Heating {units}"].sum()
            df[f"Current {type} Hot Water {usage}"] = filtered_loads_usage_DHW.groupby(allBldg_output['Timestamp'])[f"Hot Water {units}"].sum()
            df[f"Current {type} Cooking {usage}"] = filtered_loads_usage_other.groupby(allBldg_output['Timestamp'])[f"Cooking {units}"].sum()
            df[f"Current {type} Laundry {usage}"] = filtered_loads_usage_other.groupby(allBldg_output['Timestamp'])[f"Laundry {units}"].sum()
            df[f"Current {type} Other Process {usage}"] = filtered_loads_usage_other.groupby(allBldg_output['Timestamp'])[f"Other Process {units}"].sum()
            if usage == "Electricity Use (kWh)":
                    df[f"Current {type} Cooling {usage}"] = filtered_loads_usage_cooling.groupby(allBldg_output['Timestamp'])["Cooling (kWh)"].sum()


        # Reset the index to move timestamp back as a column and reset to default integer index
        df.reset_index(inplace=True)
        df["Ambient Air Wet Bulb Temp (F)"] = weather_df["Wet Bulb Temp (°F)"]
        return df
    
    current_district_therm_loads=filtered_bldg(building_meta_df,allBldgLoads_output,meets_district_cooling_conditions,meets_district_heating_conditions,meets_district_DHW_conditions,meets_district_DHW_conditions,current_district_therm_loads, "District","Load (kBtu)","(kBtu)")



    ################# bldg loads and usage #####################
    

    # empty frame
    current_bldg_therm_loads = pd.DataFrame()
    current_bldg_elec_use = pd.DataFrame()
    current_bldg_gas_use = pd.DataFrame()
    new_bldg_therm_loads = pd.DataFrame()


    ########### building conditions #########################
    # Define a function to check conditions for each building
    def meets_building_cooling_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year < row["Year of Decarb"] and
            row["Current District Cooling Y/N?"] == "N"
        )
        return condition
    
    # Define a function to check conditions for each building
    def meets_building_heating_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year < row["Year of Decarb"] and
            row["Current District Heating Y/N?"] == "N"
        )
        return condition
    
        # Define a function to check conditions for each building
    def meets_building_DHW_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year < row["Year of Decarb"] and
            row["Current District Hot Water Y/N?"] == "N"
        )
        return condition
    
    def meets_building_other_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year < row["Year of Cooking and Laundry Decarb"] 
        )
        return condition
        
    print("allBldgGasUse_output",allBldgGasUse_output)
    current_Bldg_Therm_Op_Loads=filtered_bldg(building_meta_df,allBldgLoads_output,meets_building_cooling_conditions,meets_building_heating_conditions,meets_building_DHW_conditions,meets_building_other_conditions,current_bldg_therm_loads,"Building Equipment","Load (kBtu)","(kBtu)")
    Current_Bldg_Equip_Elec_Use=filtered_bldg(building_meta_df,allBldgElecUse_output,meets_building_cooling_conditions,meets_building_heating_conditions,meets_building_DHW_conditions,meets_building_other_conditions,current_bldg_elec_use, "Building Equipment","Electricity Use (kWh)","(kWh)")
    Current_Bldg_Equip_Gas_Use=filtered_bldg(building_meta_df,allBldgGasUse_output,meets_building_cooling_conditions,meets_building_heating_conditions,meets_building_DHW_conditions,meets_building_other_conditions,current_bldg_gas_use, "Building Equipment","Gas Use (therms)","(Therms)")

    # Define a function to check conditions for each building
    def meets_new_building_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year >= row["Year of Decarb"] and
            row["Decarbonization Type (New Building Heat Pumps vs. New CUP)"] == "New Building Heat Pumps"
        )
        return condition


    def meets_new_building_other_conditions(row):
        condition = (
            year >= row["First Year Active"] and
            year <= row["Last Year Active"] and
            year >= row["Year of Cooking and Laundry Decarb"] 
        )
        return condition
    
    new_Bldg_Therm_Op_Loads=filtered_bldg(building_meta_df,allBldgLoads_output,meets_new_building_conditions,meets_new_building_conditions,meets_new_building_conditions,meets_new_building_other_conditions,new_bldg_therm_loads,"New Building Independent Heat Pump ","Load (kBtu)","(kBtu)")




    ######################## wet bulb / COP regression #######################

    calculation_map_COP_wetbulb = pd.read_excel("UCSB Calculation Map.xlsx",sheet_name="Reg. Data - Current District",header =1)


    def COP_wetbulb_reg(COP_wetbulb_reg_output,subset_training_data):
            # Define conditions for COP calculation based on wet bulb temperature
            COP_wetbulb_reg_output["Current District Cooling COP"] = np.where(
                (COP_wetbulb_reg_output["Current District Wet Bulb (F)"] <= 68),
                6,
                np.where(
                    COP_wetbulb_reg_output["Current District Wet Bulb (F)"] >= 78,
                    3,
                    np.nan  # Placeholder for values between 68 and 78 (handled in regression)
                )
            )

            # Filter the range of wet bulb temperatures for regression (between 68 and 78)
            mask = (COP_wetbulb_reg_output["Current District Wet Bulb (F)"] > 68) & (
                COP_wetbulb_reg_output["Current District Wet Bulb (F)"] < 78)
            

            wet_bulb_temps_subset = COP_wetbulb_reg_output.loc[mask, "Current District Wet Bulb (F)"]

            if not wet_bulb_temps_subset.empty:
                # Prepare data for regression (subset from calculation map)
                X = sm.add_constant(subset_training_data['Current District Wet Bulb (F)'])
                y = subset_training_data['Current District Cooling COP']

                # Fit OLS regression model
                model = sm.OLS(y, X).fit()

                # Generate a range of Wet Bulb Temperature values for prediction
                new_wet_bulb_temps = np.arange(wet_bulb_temps_subset.min(), wet_bulb_temps_subset.max() + 1)
                
                # Add constant term to the new values for prediction
                X_new = sm.add_constant(new_wet_bulb_temps)
                # print("jjjjjjjjj",new_wet_bulb_temps,X_new)
                # Predict Cooling COP for the new values using the fitted model
                predicted_cooling_cop_new = model.predict(X_new)

                # Update output DataFrame with predicted COP values
                COP_wetbulb_reg_output.loc[mask, "Current District Cooling COP"] = predicted_cooling_cop_new

                # Display model summary
                # print(model.summary())

            return COP_wetbulb_reg_output

    
    # Subset the training data based on wet bulb temperature range
    subset_training_data = calculation_map_COP_wetbulb[
        (calculation_map_COP_wetbulb['Current District Wet Bulb (F)'] > 67) &
        (calculation_map_COP_wetbulb['Current District Wet Bulb (F)'] < 79)
    ]



    COP_wetbulb_reg_output = pd.DataFrame()
    COP_wetbulb_reg_output["Current District Wet Bulb (F)"] = range(24,91)

    # Apply the COP regression function to calculate Cooling COP
    COP_wetbulb_reg_output = COP_wetbulb_reg(COP_wetbulb_reg_output,subset_training_data)



    ##################### Current District Elec use ###############################

    #empty frame
    current_District_Elec_Use = pd.DataFrame()
    # Round and convert to integer for temperature comparison
    rounded_weather_temps = weather_df["Wet Bulb Temp (°F)"].round().astype(int)
    wet_bulb_temps = COP_wetbulb_reg_output["Current District Wet Bulb (F)"]

    # Find the index of the closest wet bulb temperature for each weather temperature
    closest_indices = np.abs(wet_bulb_temps.values[:, None] - rounded_weather_temps.values).argmin(axis=0)

    # Use the closest indices to retrieve corresponding COP values
    assigned_cop_values = COP_wetbulb_reg_output["Current District Cooling COP"].iloc[closest_indices].reset_index(drop=True)

    # Assign the assigned COP values to the original DataFrame
    current_District_Elec_Use["Current District Cooling COP_WB"] = assigned_cop_values


    current_District_Elec_Use["Current District System Cooling Electricity Use (kWh)"] = current_district_therm_loads["Current District Cooling Load (kBtu)"]/current_District_Elec_Use["Current District Cooling COP_WB"]



   ############################ Current District Gas use ###############################

    current_District_Gas_Use= pd.DataFrame()

    district_HW_COP = calculation_map_COP_wetbulb['Current District Heating COP'][0]
    district_DHW_COP =calculation_map_COP_wetbulb['Current District Hot Water COP'][0]

    current_District_Gas_Use["Current District System Heating Gas Use (therms)"] = current_district_therm_loads["Current District Heating Load (kBtu)"]/district_HW_COP
    current_District_Gas_Use["Current District System Hot Water Gas Use (therms)"] = current_district_therm_loads["Current District Hot Water Load (kBtu)"]/district_DHW_COP



######################## new bldg regression #######################

    calculation_map_hpCOP_drybulb = pd.read_excel("UCSB Calculation Map.xlsx",sheet_name="Reg. Data - New Bldg Equip",header =1)


    def COP_dryBulb_newEquip_reg(COP_newBldgEquip_reg_output,subset_training_data):


                # Prepare data for regression (subset from calculation map)
                X = sm.add_constant(subset_training_data['Ambient Air Dry Bulb Temp (F)'])
                y = subset_training_data['New Building Independent Heat Pump - Cooling COP (44°F CHWST)']

                # Fit OLS regression model
                model = sm.OLS(y, X).fit()

                # Generate a range of Wet Bulb Temperature values for prediction
                new_dry_bulb_temps = np.arange(subset_training_data['Ambient Air Dry Bulb Temp (F)'].min(), subset_training_data['Ambient Air Dry Bulb Temp (F)'].max() + 1)
                print("new_dry_bulb_temps",new_dry_bulb_temps)
                # Add constant term to the new values for prediction
                X_new = sm.add_constant(new_dry_bulb_temps)
                print("X_new",X_new)
                # print("jjjjjjjjj",new_wet_bulb_temps,X_new)
                # Predict Cooling COP for the new values using the fitted model
                predicted_cooling_cop_new = model.predict(X_new)
                print("predicted_cooling_cop_new",predicted_cooling_cop_new)

                # Update output DataFrame with predicted COP values
                COP_newBldgEquip_reg_output["New Building Independent Heat Pump - Cooling COP (44°F CHWST)"] = predicted_cooling_cop_new

                # Display model summary
                # print(model.summary())

                return COP_newBldgEquip_reg_output

    
    # Subset the training data based on wet bulb temperature range
    subset_training_data_hpCOP = calculation_map_hpCOP_drybulb[(calculation_map_hpCOP_drybulb['Ambient Air Dry Bulb Temp (F)'] > 49)]
    # Remove rows with missing values from subset_training_data
    # Drop rows with NaN in the specified column only
    subset_training_data_hpCOP_cleaned = subset_training_data_hpCOP.dropna(subset=['New Building Independent Heat Pump - Cooling COP (44°F CHWST)'])
    print("subset_training_data_hpCOP",subset_training_data_hpCOP_cleaned)

    COP_newBldgEquip_reg_output = pd.DataFrame()
    COP_newBldgEquip_reg_output["Ambient Air Dry Bulb Temp (F)"] = range(50,101)

    # Apply the COP regression function to calculate Cooling COP
    COP_newBldgEquip_reg_output = COP_dryBulb_newEquip_reg(COP_newBldgEquip_reg_output,subset_training_data_hpCOP_cleaned)
















































