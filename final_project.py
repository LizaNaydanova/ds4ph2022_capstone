from dash import Dash, dcc, html, Input, Output
import dash
import dash_bootstrap_components as dbc
import os
import pandas as pd
import numpy as np
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score


#bs = 'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

#To change theme, change "UNITED" to any option from this website: https://bootswatch.com/united/
app.layout = html.Div([
    html.H1("Can we predict mental illness from demographic data and a additional questions? Enter your answers to see the the chances you have a mental illness", style={'font-size': '40px'}),
    html.Div([
        html.H2('Enter your age', style={'font-size': '20px'}),
        dcc.Input(id = 'age', value = 20, type = 'number',style={'backgroundColor': 
                   '#FFF2CC','width':'100px','margin-bottom':'20px'}),
        html.H2('Enter your gender', style={'font-size': '20px'}),
        dcc.RadioItems(options = [{'label': 'Male', 'value': '0'},{'label': 'Female', 'value': '1'}],
                       value = '0',
                       id = 'gender', style={'font-size': '15px','margin-bottom':'20px'}),
        html.H2('Enter your region', style={'font-size': '20px'}),
        dcc.Dropdown(id = 'region', options = [{'label':'East North Central','value':'0'},{'label':'East South Central','value':'1'},{'label':'Middle Atlantic','value':'2'},{'label':'Mountain','value':'3'},{'label':'New England','value':'4'},{'label':'Pacific','value':'5'},{'label':'South Atlantic','value':'6'},{'label':'West North Central','value':'7'},{'label':'West South Central','value':'8'}], value = '0', style={'width': '210px','backgroundColor': 
                   '#FFF2CC','margin-bottom':'20px'}),
        html.H2('Enter your education level', style={'font-size': '20px'}),
        dcc.Dropdown(id = 'education', options = [{'label':'Some Highschool','value':'0'},{'label':'Completed Highschool/GED','value':'1'},{'label':'Some undergraduate','value':'2'},{'label':'Completed Undergraduate','value':'3'},{'label':'Some Masters','value':'4'},{'label':'Completed Masters','value':'5'}, {'label':'Some PhD','value':'6'},{'label':'Completed PhD','value':'7'}], value = '0', style={'width': '210px','backgroundColor': 
                   '#FFF2CC','margin-bottom':'20px'}),
        html.H2('Are you currently employed at least part-time?', style={'font-size': '20px'}),
        dcc.RadioItems(options = [{'label': 'Yes', 'value': '0'},{'label': 'No', 'value': '1'}],
                       value = '0',
                       id = 'employment',style={'margin-bottom':'20px'}),
        html.H2('What is your annual income (including any social welfare programs) in thousands of USD? (example: for 40,000 per yer, enter 40)', style={'font-size': '20px'}),
        dcc.Input(id = 'income', value = 20, type = 'number',style={'font-size': '15px', 'backgroundColor':'#FFF2CC', 'width': '100px','margin-bottom':'20px'}),
        html.H2('What is the total length of any gaps in my resume in months?', style={'font-size': '20px'}),
        dcc.Input(id = 'resume_gaps', value = 20, type = 'number',style={'font-size': '15px', 'backgroundColor':'#FFF2CC', 'width': '100px','margin-bottom':'20px'}),
        html.H2('Are you legally disabled?', style={'font-size': '20px'}),
        dcc.RadioItems(options = [{'label': 'Yes', 'value': '0'},{'label': 'No', 'value': '1'}],
                       value = '0',
                       id = 'disability',style={'margin-bottom':'20px'}),
        html.H2('Do you currently live with your parents?', style={'font-size': '20px'}),
        dcc.RadioItems(options = [{'label': 'Yes', 'value': '0'},{'label': 'No', 'value': '1'}],
                       value = '0',
                       id = 'parents',style={'margin-bottom':'20px'}),
        html.H2('Do you read outside of work and school?', style={'font-size': '20px'}),
        dcc.RadioItems(options = [{'label': 'Yes', 'value': '0'},{'label': 'No', 'value': '1'}],
                       value = '0',
                       id = 'reading')
        
    ]),
    html.Br(),
    html.H1("Based on your answers, your predicted chances of having a mental illness are: ", style={'font-size': '40px'}),
    html.H1(id = 'chances'),

])




@app.callback(
    Output(component_id = 'chances'   , component_property = 'children'),
    Input(component_id  = 'age', component_property = 'value'),
    Input(component_id  = 'gender'   , component_property = 'value'),
    Input(component_id  = 'region', component_property = 'value'),
    Input(component_id  = 'education'   , component_property = 'value'),
    Input(component_id  = 'employment'   , component_property = 'value'),
    Input(component_id  = 'income'   , component_property = 'value'),
    Input(component_id  = 'resume_gaps'   , component_property = 'value'),
    Input(component_id  = 'disability'   , component_property = 'value'),
    Input(component_id  = 'parents'   , component_property = 'value'),
    Input(component_id  = 'reading'   , component_property = 'value')
)
def update_output_div(age, gender, region, education, employment, income, resume_gaps, disability, parents, reading):
    d = {'age': age, 'gender': gender, 'region':region, 'education':education, 'employment': employment, 'income':income, 'resume_gaps':resume_gaps, 'disability':disability, 'parents':parents, 'reading':reading}
    df = pd.DataFrame(data=d)
    ss = StandardScaler()
    X = ss.fit_transform(df)
    lr = train_model()
    y_pred=lr.predict(X)
    return(age, ' ', gender,' ', region, ' ', education, ' ', income,' ', resume_gaps,' ', disability,' ', parents,' ', reading, ' ', y_pred)

def train_model():
    
    df = pd.read_csv("data_new.csv")
    
    df = df.iloc[: , 1:]
    
    y = df.mental_illness
    X = df.drop(columns = ['mental_illness'])
    
    # Standardize data
    ss = StandardScaler()
    X = ss.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    
    return lr




if __name__ == '__main__':
    app.run_server(debug = True, host='jupyter.biostat.jhsph.edu', port = os.getuid() + 30)