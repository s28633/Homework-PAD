
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#
# Load dataset and extract first 10 rows.
#
df = pd.read_csv('C:\\Users\\Antek\\Desktop\\pad6\\homework\\Task2\\winequelity.csv')
df_head = df.head(10)

#
# Select all columns but 'pH'(main comparision value), 'Unnamed: 0'(indices) and 'target' (non-numerical data).
#
df_no_ph_col = df.loc[:, df.columns != 'pH']
df_no_ph_col = df_no_ph_col.loc[:, df_no_ph_col.columns != 'Unnamed: 0']
df_no_ph_col = df_no_ph_col.loc[:, df_no_ph_col.columns != 'target']

#
# Declare layout objects.
#
dash_text_data_header = html.Div([html.H4('Header of winequelity.csv dataset')])
dash_table_data_header =  dash_table.DataTable(df_head.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
dash_text_alg_selection = html.Div([html.H4('Please select the algorithm:')])
dash_dropdown_alg_selection = dcc.Dropdown(['Linear Regression', 'Classification'], 'Linear Regression', id='operation-type-dropdown')
dash_text_atrr_to_compare = html.Div([html.H4('Please select attribute to compare with:')])
dash_dropdown_atrr_to_compare = dcc.Dropdown(df_no_ph_col.columns, 'fixed acidity', id='attr-to-comp-dropdown')
dash_graph = dcc.Graph(id='data-graph')

#
# Spawn app object and compose its layout.
#
app = Dash(__name__)
app.layout = html.Div(
    [
        dash_text_data_header,
        dash_table_data_header,
        dash_text_alg_selection,
        dash_dropdown_alg_selection,
        dash_text_atrr_to_compare,
        dash_dropdown_atrr_to_compare,
        dash_graph
    ]
)

#
# Create graph's callback
#
@app.callback(Output('data-graph', 'figure'), 
              [Input("operation-type-dropdown", "value"), Input('attr-to-comp-dropdown', 'value')])
def update_figure(algorithm, attribute):

    if algorithm == "Linear Regression":
        #
        # First draw all of the points in data set composed
        # of (selected_value, pH entry)
        #
        fig = px.scatter(df, x=attribute, y="pH", opacity=0.2)

        #
        # Next, we build and train the model so that 'selected_value'
        # will be used for predicting 'pH' values.
        #
        X = df[attribute].values.reshape(-1, 1)
        Y = df["pH"]
        model = LinearRegression().fit(X,Y)

        #
        # Create a line that visualizes linear regression fit.
        # Line spawns x coordinate based on min/max values of selected attribute
        # and y coordinate based on model prediction from the x values.
        #
        line_x = np.linspace(X.min(), X.max(), 100)
        line_y = model.predict(line_x.reshape(-1, 1))
        fig.add_traces(go.Scatter(x=line_x, y=line_y, name='Regression Fit'))
        
        return fig

    else:
        #
        # Split the data into test and train sets.
        # 
        train, test = train_test_split(df, test_size=0.2)

        #
        # Train Naive Bayes classifier on training set.
        #
        X_train = train[attribute].values.reshape(-1, 1)
        y_train = train.target
        classificator = GaussianNB().fit(X_train, y_train)

        #
        # Used trained model on testing set.
        #
        X_test = test[attribute].values.reshape(-1, 1)
        y_test = test.target
        y_pred = classificator.predict(X_test)

        #
        # Calculate and normalize confusion matrix.
        #
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #
        # Prepare confusion matrix figure.
        #
        labels = ["Red Wine", "White Wine"]
        fig = ff.create_annotated_heatmap(cm, x=labels, y=labels, colorscale='Viridis')
        fig['data'][0]['showscale'] = True
        fig.add_annotation(dict(font=dict(color="black",size=14), x=0.5, y=-0.15, showarrow=False,
                                text="Predicted value", xref="paper", yref="paper"))

        return fig

#
# Run the app.
#
if __name__ == '__main__':
    app.run_server(debug=True)