from pydantic import BaseModel

# 2. Class which describes Bank Notes measurements


class CensusData(BaseModel):
    age: float
    fnlgt: float
    education_num: float
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    # x0_Divorced: str
    # x0_Married_AF_spouse: str
    # x0_Married_civ_spouse: str
    # x0_Married_spouse_absent: str
    # x0_Never_married: str
    # x0_Separated: str
    # x0_Widowed: str
    # x1_Amer_Indian_Eskimo: str
    # x1_Asian_Pac_Islander: str
    # x1_Black: str
    # x1_Other: str
    # x1_White: str
    # x2_Husband: str
    # x2_Not_in_family: str
    # x2_Other_relative: str
    # x2_Own_child: str
    # x2_Unmarried: str
    # x2_Wife: str
    x3_Female: str
    x3_Male: str
