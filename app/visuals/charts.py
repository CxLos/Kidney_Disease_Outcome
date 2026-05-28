# ============================ CHARTS =============================== #

import plotly.express as px
import plotly.graph_objects as go


# ========================== Model Performance Metrics ========================== #

def build_metrics_bar(df_metrics):
    df_long = df_metrics.melt(
        id_vars='Model',
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard'],
        var_name='Metric',
        value_name='Score'
    )

    fig = px.bar(
        df_long,
        x='Score',
        y='Metric',
        color='Model',
        barmode='group',
        orientation='h',
        text=df_long['Score'].apply(lambda x: f"{x:.2%}"),
    )

    fig.update_layout(
        height=600,
        width=800,
        title=dict(
            text='Model Performance Metrics',
            x=0.5,
            font=dict(size=25, family='Calibri', color='black')
        ),
        font=dict(family='Calibri', size=18, color='black'),
        xaxis=dict(
            title=dict(text='Score', font=dict(size=20)),
            range=[0, 1],
            tickformat='.0%',
        ),
        yaxis=dict(
            title=dict(text='Metric', font=dict(size=20)),
            tickfont=dict(size=18)
        ),
        legend=dict(
            title_text='Model',
            orientation='v',
            x=1.05, y=1,
            xanchor='left', yanchor='top',
        ),
        hovermode='closest',
        bargap=0.15,
        bargroupgap=0.05,
    )

    fig.update_traces(
        textposition='auto',
        hovertemplate='<b>Model:</b> %{color}<br><b>Metric:</b> %{y}<br><b>Score:</b> %{text}<extra></extra>'
    )

    return fig


# ========================== CKD Status Distribution ========================== #

def build_ckd_bar(df):
    df_ckd_status = df['CKD_Status'].value_counts().reset_index()
    df_ckd_status.columns = ['CKD Status', 'Count']

    fig = px.bar(
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
            font=dict(size=25, family='Calibri', color='black')
        ),
        font=dict(family='Calibri', size=18, color='black'),
        xaxis=dict(
            tickangle=0,
            tickfont=dict(size=18),
            title=dict(text='CKD Status', font=dict(size=20)),
            showticklabels=True
        ),
        yaxis=dict(
            title=dict(text='Count', font=dict(size=20)),
        ),
        legend=dict(
            title_text='',
            orientation='v',
            x=1.05, y=1,
            xanchor='left', yanchor='top',
            visible=True
        ),
        hovermode='closest',
        bargap=0.08,
        bargroupgap=0,
    ).update_traces(
        textposition='auto',
        hovertemplate='<b>Status:</b> %{x}<br><b>Count</b>: %{y}<extra></extra>'
    )

    return fig


# =========================== Confusion Matrix ========================== #

def build_confusion_matrix(cm):
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x='Predicted', y='Actual', color='Count'),
        x=['No CKD', 'CKD'],
        y=['No CKD', 'CKD'],
        title='Confusion Matrix (Random Forest)',
        color_continuous_scale='blues',
    )
    fig.update_layout(
        font=dict(family='Calibri', size=16),
        title_x=0.5,
        width=700,
        height=600
    )
    return fig


# =========================== Feature Importance ========================== #

def build_feature_importance(feature_importance_df):
    fig = px.bar(
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
    return fig


# =========================== Probability Score Distribution ========================== #

def build_score_dist(y_proba):
    fig = px.histogram(
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
        marker=dict(line=dict(width=1, color='black'))
    )
    return fig


# =========================== ROC Curve ========================== #

def build_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title='ROC Curve (Random Forest)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title_x=0.5,
        width=800,
        height=600,
        font=dict(family='Calibri', size=18)
    )
    return fig


# =========================== Data Table ========================== #

def build_data_table(df):
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='center',
            height=30,
            font=dict(size=12)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            height=25,
            font=dict(size=12)
        )
    )])

    fig.update_layout(
        margin=dict(l=50, r=50, t=30, b=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig
