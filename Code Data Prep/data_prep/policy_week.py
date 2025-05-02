import pandas as pd


def get_policy_week():
    import pandas as pd
    state_action = pd.read_csv("../Data/State_Action_Spreadsheet_Clean_3.0.csv")
    state_action.State = state_action.State.map(lambda x: x.upper() if type(x) == str else x)
    state_action.drop([50,51, 52,53,54], inplace=True)
    #state_action.loc[4, "Action_Date"] = "4/1/2020"
    #state_action.loc[20, "Action_Date"] = "3/25/2020"
    #state_action.loc[25, "Action_Date"] = "3/25/2020"
    state_action= state_action[["State","Encoded","Reverse_Date", "Action_Date"]] 
    state_action["Reverse_Date"] = state_action["Reverse_Date"].fillna("1/2/2022")
    state_action["Action_Date"] = state_action["Action_Date"].fillna("1/2/2022")
    state_action.rename(columns={"Encoded":"policy","State":"state"}, inplace=True)
    print("Updated Reversal Date")
    return state_action