# =================================== IMPORTS ================================= #

# ------ System Imports ------ #
import os
import sys

# ------ Python Imports ------ #
# import csv, sqlite3
import numpy as np 
import pandas as pd 
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter

# ------ Machine Learning Imports ------ #
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.pipeline import Pipeline

# ------ Dash Imports ------ #
import dash
from dash import dcc, html
# from dash.dependencies import Input, Output, State

# -------------------------------------- DATA ------------------------------------------- #

current_dir = os.getcwd()
current_file = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Current Directory: {current_dir}")
# print(f"Current File: {current_file}")
# print(f"Script Directory: {script_dir}")
# con = sqlite3.connect("Chicago_Schools.db")
# cur = con.cursor()

# file = r'c:\Users\CxLos\OneDrive\Documents\Portfolio Projects\Machine Learning\Kidney_Disease_Outcome\data\kidney_disease_dataset.xlsx'

file = os.path.join(script_dir, 'data', 'kidney_disease_dataset.xlsx')

df = pd.read_excel(file)

# print(df.head(10))
print(f'DF Shape: \n {df.shape}')
# print(f'Number of rows: {df.shape[0]}')
# print(f'Column names: \n {df.columns}')
# print(df.info())
# print(df.describe())
# print(df.dtypes)

# ================================= Columns ================================= #

columns = [
    'Age', 
    'Creatinine_Level',
    'BUN',
    'Diabetes',
    'Hypertension', 
    'GFR',
    'Urine_Output', 
    'CKD_Status',
    'Dialysis_Needed'
]

# Rename columns
# df.rename(
#     columns={
#         "" : "",
#     }, 
# inplace=True)

# ============================== Data Preprocessing ========================== #

# Missing Values
missing = df.isnull().sum()
# print('Columns with missing values before fillna: \n', missing[missing > 0])

# df.fillna(df.median(numeric_only=True), inplace=True)
# df.fillna(df.mode().iloc[0], inplace=True)  # for categorical values

# Check for duplicate columns
# duplicate_columns = df.columns[df.columns.duplicated()].tolist()
# print(f"Duplicate columns found: {duplicate_columns}")
# if duplicate_columns:
#     print(f"Duplicate columns found: {duplicate_columns}")

# Encode categorical variables
# df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})
# df['Hypertension'] = df['Hypertension'].map({'Yes': 1, 'No': 0})
# df['CKD_Status'] = df['CKD_Status'].map({'Yes': 1, 'No': 0})

features = [
    'Age', 
    'Creatinine_Level',
    'BUN',
    'Diabetes',
    'Hypertension', 
    # 'GFR',
    'Urine_Output'
]
target = 'CKD_Status'

X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()

# Fit and transform the features so they have a mean of 0 and a standard deviation of 1
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, # Features
    y, # Target variable
    test_size=0.2, # 20% of the data will be used for testing
    # train_size=0.8, # 80% of the data will be used for
    random_state=42, # For reproducibility
    # shuffle=True, # Shuffle the data before splitting
    stratify=y # Ensure that the split maintains the same proportion of classes in the target variable
)

# ---------------------------- Machine Learning Models ----------------------- #

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

cv_scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)
# print("CV Accuracy scores: \n", cv_scores)
# print("Mean CV Accuracy:", np.mean(cv_scores))

# ---------------------------- Model Evaluation ----------------------- #

models = {'Logistic Regression': log_reg, 'Decision Tree': dt, 'Random Forest': rf}

metrics_summary = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    jaccard = jaccard_score(y_test, y_pred, pos_label=1)

    metrics_summary.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'Jaccard': jaccard
    })

# Convert to DataFrame for visualization or printing
df_metrics = pd.DataFrame(metrics_summary)
# print(df_metrics)

# ============================== Data Visualization ========================== #

# Set default theme for seaborn
# sns.set_theme(style="whitegrid")
# # Set default theme for plotly
# px.defaults.template = "plotly_white"
# # Set default color palette for plotly
# px.defaults.color_discrete_sequence = px.colors.qualitative.Plotly
# # Set default font for plotly
# px.defaults.font_family = "Calibri"
# # Set default font size for plotly
# px.defaults.font_size = 18

# ========================== Model Performance Metrics ========================== #

# Convert wide df_metrics to long format for bar plot
df_long = df_metrics.melt(id_vars='Model', 
                          value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard'],
                          var_name='Metric', 
                          value_name='Score')

df_long = df_metrics.melt(
    id_vars='Model',
    value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard'],
    var_name='Metric',
    value_name='Score'
)

# Create grouped horizontal bar chart
metrics_bar = px.bar(
    df_long,
    x='Score',
    y='Metric',
    color='Model',
    barmode='group',
    orientation='h',
    text=df_long['Score'].apply(lambda x: f"{x:.2%}"),
)

# Customize layout
metrics_bar.update_layout(
    height=600,
    width=800,
    title=dict(
        text='Model Performance Metrics',
        x=0.5,
        font=dict(
            size=25,
            family='Calibri',
            color='black',
        )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        title=dict(
            text='Score',
            font=dict(size=20),
        ),
        range=[0, 1],
        tickformat='.0%',
    ),
    yaxis=dict(
        title=dict(
            text='Metric',
            font=dict(size=20),
        ),
        tickfont=dict(size=18)
    ),
    legend=dict(
        title_text='Model',
        orientation="v",
        x=1.05,
        y=1,
        xanchor="left",
        yanchor="top",
    ),
    hovermode='closest',
    bargap=0.15,
    bargroupgap=0.05,
)

metrics_bar.update_traces(
    textposition='auto',
    hovertemplate='<b>Model:</b> %{color}<br><b>Metric:</b> %{y}<br><b>Score:</b> %{text}<extra></extra>'
)

# ---------------------------- CKD Status Distribution ----------------------- #

# df_activity_status = df.groupby('Activity Status').size().reset_index(name='Count')

 # Example: value counts of CKD_Status
df['CKD_Status'] = df['CKD_Status'].astype(str)  # Ensure CKD_Status is string type
df_ckd_status = df['CKD_Status'].value_counts().reset_index()
df_ckd_status.columns = ['CKD Status', 'Count']

ckd_bar = px.bar(
    df_ckd_status,
    x='CKD Status',
    y='Count',
    color='CKD Status',
    text='Count',
).update_layout(
    height=600, 
    width=800,
    title=dict(
        text='CKD Status Distribution',
        x=0.5, 
        font=dict(
            size=25,
            family='Calibri',
            color='black',
        )
    ),
    font=dict(
        family='Calibri',
        size=18,
        color='black'
    ),
    xaxis=dict(
        tickangle=0,
        tickfont=dict(size=18),
        title=dict(
            text="CKD Status",
            font=dict(size=20),
        ),
        showticklabels=True
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=20),
        ),
    ),
    legend=dict(
        title_text='',
        orientation="v",
        x=1.05,
        y=1,
        xanchor="left",
        yanchor="top",
        visible=True
    ),
    hovermode='closest',
    bargap=0.08,
    bargroupgap=0,
).update_traces(
    textposition='auto',
    hovertemplate='<b>Status:</b> %{x}<br><b>Count</b>: %{y}<extra></extra>'
)

# =========================== Confusion Matrix ========================== #

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig_cm = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=['No CKD', 'CKD'],
    y=['No CKD', 'CKD'],
    title="Confusion Matrix (Random Forest)",
    color_continuous_scale='blues',
)
fig_cm.update_layout(
    font=dict(family='Calibri', size=16),
    title_x=0.5,
    width=700,
    height=600
)

# =========================== Feature Importance ========================== #

importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

fig_fi = px.bar(
    feature_importance_df,
    x='Feature',
    y='Importance',
    title='Feature Importance (Random Forest)',
    text='Importance',
    color='Feature'
).update_layout(
    height=600,
    width=800,
    title_x=0.5,
    font=dict(family='Calibri', size=18),
    xaxis_title='Feature',
    yaxis_title='Importance',
    showlegend=False
).update_traces(
    texttemplate='%{text:.3f}',
    textposition='outside'
)

# =========================== Probability Score Distributions ========================== #

y_proba = rf.predict_proba(X_test)[:, 1]  # Prob for CKD=1

fig_score_dist = px.histogram(
    x=y_proba,
    nbins=30,
    title='Prediction Probability Distribution (CKD)',
    labels={'x': 'Probability of CKD', 'y': 'Count'},
    opacity=0.8
).update_layout(
    title_x=0.5,
    font=dict(family='Calibri', size=18),
    bargap=0,
    width=800,
    height=600,
).update_traces(
    marker=dict(
        line=dict(
            width=1,         # Thickness of the outline
            color='black'    # Color of the outline
        )
    )
)

# =========================== ROC Curve ========================== #

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.2f}"))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(
    title='ROC Curve (Random Forest)',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    title_x=0.5,
    width=800,
    height=600,
    font=dict(family='Calibri', size=18)
)

# # ========================== DataFrame Table ========================== #

df_table = go.Figure(data=[go.Table(
    # columnwidth=[50, 50, 50],  # Adjust the width of the columns
    header=dict(
        values=list(df.columns),
        fill_color='paleturquoise',
        align='center',
        height=30,  # Adjust the height of the header cells
        # line=dict(color='black', width=1),  # Add border to header cells
        font=dict(size=12)  # Adjust font size
    ),
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='lavender',
        align='left',
        height=25,  # Adjust the height of the cells
        # line=dict(color='black', width=1),  # Add border to cells
        font=dict(size=12)  # Adjust font size
    )
)])

df_table.update_layout(
    margin=dict(l=50, r=50, t=30, b=40),  # Remove margins
    height=400,
    # width=1500,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# ============================== Dash Application ========================== #

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    children=[ 
        html.Div(
            className='divv', 
            children=[ 
                html.H1('Predicting Kidney Disease Outcome', className='title'),
                # html.H1(f'', className='title2'),
                html.Div(
                    className='btn-box', 
                    children=[
                        html.A(
                            'Repo',
                            href=f'https://github.com/CxLos/Kidney_Disease_Outcome',
                            className='btn'
                        )
                    ]
                )
            ]
        ),
        
        # Data Table
        html.Div(
            className='row0',
            children=[
                html.Div(
                    className='table',
                    children=[
                        html.H1(
                            className='table-title',
                            children='Kidney Disease Data Table'
                        )
                    ]
                ),
                html.Div(
                    className='table2', 
                    children=[
                        dcc.Graph(
                            className='data',
                            figure=df_table
                        )
                    ]
                )
            ]
        ),

        # html.Div(
        #     className='row1',
        #     children=[
        #         html.Div(
        #             className='graph11',
        #             children=[
        #                 html.Div(
        #                     className='high1', 
        #                     children=[f'Placeholder']),
        #                 html.Div(
        #                     className='circle1',
        #                     children=[
        #                         html.Div(
        #                             className='hilite',
        #                             children=[html.H1(
        #                                 className='high2', 
        #                                 children=[])]
        #                         )
        #                     ]
        #                 )
        #             ]
        #         ),
        #         html.Div(
        #             className='graph22',
        #             children=[
        #                 html.Div(
        #                     className='high3', 
        #                     children=[f'Placeholder:']),
        #                 html.Div(
        #                     className='circle2',
        #                     children=[
        #                         html.Div(
        #                             className='hilite',
        #                             children=[html.H1(
        #                                 className='high4', 
        #                                 children=[])]
        #                         )
        #                     ]
        #                 ) 
        #             ]
        #         )
        #     ]
        # ),

        html.Div(
            className='row1',
            children=[
                html.Div(
                    className='graph1',
                    children=[
                        dcc.Graph(
                            figure=ckd_bar
                        )
                    ]
                ),
                html.Div(
                    className='graph2',
                    children=[
                        dcc.Graph(
                            figure=fig_cm
                        )
                    ]
                )
            ]
        ),

        html.Div(
            className='row1',
            children=[
                html.Div(
                    className='graph1',
                    children=[
                        dcc.Graph(
                            figure=fig_fi
                        )
                    ]
                ),
                html.Div(
                    className='graph2',
                    children=[
                        dcc.Graph(
                            figure=fig_score_dist
                        )
                    ]
                )
            ]
        ),

        html.Div(
            className='row1',
            children=[
                html.Div(
                    className='graph1',
                    children=[
                        dcc.Graph(
                            figure=fig_roc
                        )
                    ]
                ),
                html.Div(
                    className='graph2',
                    children=[
                        dcc.Graph(
                            figure=metrics_bar
                        )
                    ]
                )
            ]
        ),
])

print(f"Serving Flask app '{current_file}'! 🚀")

if __name__ == '__main__':
    app.run_server(debug=
                    True)
                    # False)
# =================================== Updated Database ================================= #

# updated_path = f'data/kidney_disease_outcome_cleaned.xlsx'.xlsx'
# data_path = os.path.join(script_dir, updated_path)

# with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
#     df.to_excel(
#             writer, 
#             sheet_name=f'Engagement {current_month} {report_year}', 
#             startrow=1, 
#             index=False
#         )

#     # Access the workbook and each worksheet
#     workbook = writer.book
#     sheet1 = writer.sheets['Kidney Disease Outcome']
    
#     # Define the header format
#     header_format = workbook.add_format({
#         'bold': True, 
#         'font_size': 13, 
#         'align': 'center', 
#         'valign': 'vcenter',
#         'border': 1, 
#         'font_color': 'black', 
#         'bg_color': '#B7B7B7',
#     })
    
#     # Set column A (Name) to be left-aligned, and B-E to be right-aligned
#     left_align_format = workbook.add_format({
#         'align': 'left',  # Left-align for column A
#         'valign': 'vcenter',  # Vertically center
#         'border': 0  # No border for individual cells
#     })

#     right_align_format = workbook.add_format({
#         'align': 'right',  # Right-align for columns B-E
#         'valign': 'vcenter',  # Vertically center
#         'border': 0  # No border for individual cells
#     })
    
#     # Create border around the entire table
#     border_format = workbook.add_format({
#         'border': 1,  # Add border to all sides
#         'border_color': 'black',  # Set border color to black
#         'align': 'center',  # Center-align text
#         'valign': 'vcenter',  # Vertically center text
#         'font_size': 12,  # Set font size
#         'font_color': 'black',  # Set font color to black
#         'bg_color': '#FFFFFF'  # Set background color to white
#     })

#     # Merge and format the first row (A1:E1) for each sheet
#     sheet1.merge_range('A1:N1', f'Engagement Report {current_month} {report_year}', header_format)

#     # Set column alignment and width
#     # sheet1.set_column('A:A', 20, left_align_format)   

#     print(f"Kidney Disease Excel file saved to {data_path}")

# -------------------------------------------- KILL PORT ---------------------------------------------------

# netstat -ano | findstr :8050
# taskkill /PID 24772 /F
# npx kill-port 8050

# ---------------------------------------------- Host Application -------------------------------------------

# 1. pip freeze > requirements.txt
# 2. add this to procfile: 'web: gunicorn kidney_disease:server'
# 3. heroku login
# 4. heroku create
# 5. git push heroku main

# Create venv 
# virtualenv venv 
# source venv/bin/activate # uses the virtualenv

# Update PIP Setup Tools:
# pip install --upgrade pip setuptools

# Install all dependencies in the requirements file:
# pip install -r requirements.txt

# Check dependency tree:
# pipdeptree
# pip show package-name

# Remove:
# pypiwin32
# pywin32
# jupytercore
# ipykernel
# ipython

# Add:
# gunicorn==22.0.0

# ----------------------------------------------------

# Name must start with a letter, end with a letter or digit and can only contain lowercase letters, digits, and dashes.

# Heroku Setup:
# heroku login
# heroku create kidney-disease-outcome
# heroku git:remote -a kidney-disease-outcome
# git push heroku main

# Clear Heroku Cache:
# heroku plugins:install heroku-repo
# heroku repo:purge_cache -a mc-impact-11-2024

# Set buildpack for heroku
# heroku buildpacks:set heroku/python

# Heatmap Colorscale colors -----------------------------------------------------------------------------

#   ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
            #  'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
            #  'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
            #  'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
            #  'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
            #  'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
            #  'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
            #  'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
            #  'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
            #  'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
            #  'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
            #  'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
            #  'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
            #  'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
            #  'ylorrd'].

# rm -rf ~$bmhc_data_2024_cleaned.xlsx
# rm -rf ~$bmhc_data_2024.xlsx
# rm -rf ~$bmhc_q4_2024_cleaned2.xlsx