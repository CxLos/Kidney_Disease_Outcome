# ============================= LAYOUT ============================== #

from dash import dcc, html

from app.data.loader import load_data, preprocess_data
from app.models.train import train_models, evaluate_models, get_rf_artifacts
from app.visuals.charts import (
    build_metrics_bar,
    build_ckd_bar,
    build_confusion_matrix,
    build_feature_importance,
    build_score_dist,
    build_roc_curve,
    build_data_table,
)


def create_layout():
    df = load_data()
    X_scaled, y, X_train, X_test, y_train, y_test = preprocess_data(df)

    models = train_models(X_train, y_train)
    df_metrics = evaluate_models(models, X_test, y_test)
    rf_artifacts = get_rf_artifacts(models['Random Forest'], X_scaled, y, X_test, y_test)

    # Build figures
    df_table     = build_data_table(df)
    ckd_bar      = build_ckd_bar(df)
    fig_cm       = build_confusion_matrix(rf_artifacts['cm'])
    fig_fi       = build_feature_importance(rf_artifacts['feature_importance_df'])
    fig_score_dist = build_score_dist(rf_artifacts['y_proba'])
    fig_roc      = build_roc_curve(rf_artifacts['fpr'], rf_artifacts['tpr'], rf_artifacts['roc_auc'])
    metrics_bar  = build_metrics_bar(df_metrics)

    return html.Div(
        children=[
            # ------------- Header ------------- #
            html.Div(
                className='divv',
                children=[
                    html.H1('Predicting Kidney Disease Outcome', className='title'),
                    html.H1('Machine Learning Models for Clinical Risk Assessment', className='title2'),
                    html.Div(
                        className='btn-box',
                        children=[
                            html.A(
                                'Repo',
                                href='https://github.com/CxLos/Kidney_Disease_Outcome',
                                className='repo-btn'
                            )
                        ]
                    )
                ]
            ),

            # ------------- Data Table ------------- #
            html.Div(
                className='row0',
                children=[
                    html.Div(
                        className='table',
                        children=[
                            html.H1(className='table-title', children='Kidney Disease Data Table')
                        ]
                    ),
                    html.Div(
                        className='table2',
                        children=[
                            dcc.Graph(className='data', figure=df_table)
                        ]
                    )
                ]
            ),

            # ------------- Row 1: CKD Status | Confusion Matrix ------------- #
            html.Div(
                className='row1',
                children=[
                    html.Div(className='graph1', children=[dcc.Graph(figure=ckd_bar)]),
                    html.Div(className='graph2', children=[dcc.Graph(figure=fig_cm)]),
                ]
            ),

            # ------------- Row 2: Feature Importance | Score Distribution ------------- #
            html.Div(
                className='row1',
                children=[
                    html.Div(className='graph1', children=[dcc.Graph(figure=fig_fi)]),
                    html.Div(className='graph2', children=[dcc.Graph(figure=fig_score_dist)]),
                ]
            ),

            # ------------- Row 3: ROC Curve | Metrics Bar ------------- #
            html.Div(
                className='row1',
                children=[
                    html.Div(className='graph1', children=[dcc.Graph(figure=fig_roc)]),
                    html.Div(className='graph2', children=[dcc.Graph(figure=metrics_bar)]),
                ]
            ),

            # ------------- README ------------- #
            html.Div(
                className='readme-section',
                children=[
                    html.H2('📘 README'),

                    html.H4('Description'),
                    html.P(
                        'This project leverages machine learning to predict the likelihood of chronic kidney disease (CKD) '
                        'using a clinical dataset. The goal is to assist in early detection of CKD by analyzing relevant '
                        'patient biomarkers and visualizing key insights through an interactive Plotly/Dash dashboard. '
                        'The project includes preprocessing, training a model, evaluating performance, and highlighting '
                        'feature importance.'
                    ),

                    html.H4('📦 Installation'),
                    html.P('To run this project locally, follow these steps:'),
                    html.Pre([
                        html.Code(
                            'git clone https://github.com/CxLos/Kidney_Disease_Outcome\n'
                            'cd Kidney_Disease_Outcome\n'
                            'pip install -r requirements.txt'
                        )
                    ]),

                    html.H4('🧪 Methodology'),
                    html.Ul([
                        html.Li('Dataset sourced from Kaggle with 2,300+ patients\' clinical measurements.'),
                        html.Li('Preprocessing included handling missing values, outlier treatment, categorical encoding, and normalization.'),
                        html.Li('Models trained: Logistic Regression, Decision Tree, and Random Forest.'),
                        html.Li('Evaluated using accuracy, precision, recall, F1-score.'),
                        html.Li('Feature importance used to understand drivers of CKD prediction.'),
                    ]),

                    html.H4('🔍 Insights'),
                    html.Ul([
                        html.Li('Random Forest achieved the highest overall performance in accuracy and F1 score, indicating a strong balance between precision and recall.'),
                        html.Li('Decision Tree showed decent performance but slightly lagged behind Random Forest.'),
                        html.Li('Logistic Regression had the lowest scores across most metrics, making it the least effective model in this comparison.'),
                    ]),

                    html.H4('✅ Conclusion'),
                    html.P(
                        'This project demonstrates the application of machine learning for health diagnostics. '
                        'By combining statistical insights with interactive visualizations, it offers a powerful tool for '
                        'analyzing kidney disease outcomes. Future improvements could include using ensemble models or '
                        'deploying the app with live patient data integration.'
                    ),

                    html.H4('📄 License'),
                    html.P('MIT License © 2025 CxLos'),
                    html.Pre([
                        html.Code(
                            'Permission is hereby granted, free of charge, to any person obtaining a copy\n'
                            'of this software and associated documentation files (the "Software"), to deal\n'
                            'in the Software without restriction, including without limitation the rights\n'
                            'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n'
                            'copies of the Software, and to permit persons to whom the Software is\n'
                            'furnished to do so, subject to the following conditions:\n\n'
                            'The above copyright notice and this permission notice shall be\n'
                            'included in all copies or substantial portions of the Software.\n\n'
                            'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
                            'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n'
                            'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n'
                            'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n'
                            'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n'
                            'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n'
                            'SOFTWARE.'
                        )
                    ]),
                ]
            ),
        ]
    )
