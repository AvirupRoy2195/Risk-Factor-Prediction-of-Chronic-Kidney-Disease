from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
risk_factor_prediction_of_chronic_kidney_disease = fetch_ucirepo(id=857) 
  
# data (as pandas dataframes) 
X = risk_factor_prediction_of_chronic_kidney_disease.data.features 
y = risk_factor_prediction_of_chronic_kidney_disease.data.targets 
  
# metadata 
metadata_str = str(risk_factor_prediction_of_chronic_kidney_disease.metadata)
variables_str = str(risk_factor_prediction_of_chronic_kidney_disease.variables)

with open('dataset_info.txt', 'w') as f:
    f.write("--- METADATA ---\n")
    f.write(metadata_str)
    f.write("\n\n--- VARIABLE INFORMATION ---\n")
    f.write(variables_str)

print("Dataset information saved to dataset_info.txt")
