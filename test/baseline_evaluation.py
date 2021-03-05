
import pandas as pd
import plotly.graph_objects as go


from app.algo import compute_roc_auc

dir = "/home/spaethju/Datasets/Workflows/Classification/ilpd/coordinator"

fig = go.Figure()
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)


roc = pd.read_csv(dir + "/roc.csv")
fpr = roc["fpr"].tolist()
tpr = roc["tpr"].tolist()
thr = roc["thresholds"].tolist()

auc = round(compute_roc_auc(fpr, tpr), 3)

df = pd.DataFrame(data=[fpr, tpr, thr]).transpose()
df.columns = ["fpr", "tpr", "thresholds"]

fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'Federated (AUC={auc})', mode='lines'))
fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'Central (AUC={auc})', mode='lines'))

fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=500
)
fig.show()
