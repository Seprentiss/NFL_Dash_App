import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

load_figure_template('JOURNAL')
# Generate data for the distplot
np.random.seed(0)

wins = 109
loses = 55

espn_wins = 105
espn_loses = 59

# vegas_wins = 57
# vegas_loses = 30

wp = round(wins / (wins + loses) * 100, 2)
ewp = round(espn_wins / (espn_wins + espn_loses) * 100, 2)
# vwp = round(vegas_wins / (vegas_wins + vegas_loses) * 100, 2)

mean_x1 = 0  # Default mean for x1
variance_x1 = 1  # Default variance for x1
mean_x2 = 1  # Default mean for x2
variance_x2 = 1  # Default variance for x2

group_labels = ['Home Team', 'Away Team']

colors = ['slategray', 'red']

fig = ff.create_distplot(
    [np.random.normal(mean_x1, np.sqrt(variance_x1), 1000), np.random.normal(mean_x2, np.sqrt(variance_x2), 1000)],
    group_labels, bin_size=0.5, curve_type='normal', colors=colors)
fig.update_layout(title_text='Teams Performance Distribution (Now Showing Example Plot)')
fig.update_xaxes(title_text='Performance Above Average (%)')  # Add X-axis label
fig.update_yaxes(title_text='Probability')  # Add Y-axis label

# new plot

X = np.random.normal(.11, np.sqrt(.05), 1000)
Y = np.random.normal(.08, np.sqrt(.14), 1000)

diff = difference = X - Y

# Create a histogram trace
histogram_trace = go.Histogram(x=diff, histnorm='probability density',
                               name='Difference in Performance Between Home & Away', marker=dict(color='slategray'))

# Create a vertical line trace at X=0
vertical_line_trace = go.Scatter(
    x=[0, 0],
    y=[0, 1],
    mode='lines',
    name='Vertical Line at X=0 ( More to the Right is Good for the Home Team )',
    line=dict(color='red')
)

# Read data from a local CSV file
df = pd.read_csv('wp_data.csv', index_col=False)

pow_data = pd.read_csv('Pow_data.csv', index_col=False)

table_data = df.drop(
    columns=['Home_Team_DVOA', 'Home_Team_Variance', 'Home_Color', 'Home_Team_WP', 'Away_Team_DVOA',
             'Away_Team_Variance',
             "Away_Color", 'Away_Team_WP'])

graph1 = dcc.Graph(id='distplot', figure=fig),
graph2 = dcc.Graph(
    id='diffplot',
    figure={
        'data': [histogram_trace, vertical_line_trace],
        'layout': go.Layout(
            title='In Game Performance Difference Distribution (Now Showing Example Plot)',
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Frequency'),
            bargap=0.05  # Adjust the gap between bars in the histogram
        )

    }
)

sorted_categories = pow_data["Team"][::-1]
sorted_mid_values = pow_data["Mid Value"][::-1]
sorted_low_values = pow_data["Low Value"][::-1]
sorted_high_values = pow_data["High Value"][::-1]
sorted_colors = pow_data["Color"][::-1]

# Create the horizontal bar chart with sorted mid values and error bars using Plotly
figure1 = go.Figure()

for Team, mid, low, high, color in zip(sorted_categories, sorted_mid_values, sorted_low_values, sorted_high_values,
                                       sorted_colors):
    figure1.add_trace(
        go.Scatter(
            y=[Team],
            x=[mid],
            error_x=dict(
                type='data',
                symmetric=True,
                array=[(high - low) / 2]
            ),
            name=Team,
            line=dict(color=color)
        )
    )

# Customize the chart title and axis labels
figure1.update_layout(
    title='Power Rankings for All Teams',
    xaxis_title='Team Performance Value %',
    yaxis_title='Team',
    legend_traceorder="reversed"
)


# Make the chart interactive
figure1.update_xaxes(
    showspikes=True,
    spikemode='across',
    spikethickness=1,
    spikedash='dot'
)

graph3 = dcc.Graph(figure=figure1)

graph4 = dcc.Graph(
    id='pie-chart',
    figure={
        'data': [go.Pie(labels=["Home", "Away"], values=[65, 35], pull=[0.1, 0],
                        marker=dict(colors=['slategrey', 'red'], line=dict(color='#000000', width=2)),
                        hoverinfo='label+percent')],
        'layout': {
            'title': 'Example W% Pie Chart'
        }
    }
)

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY,dbc_css], suppress_callback_exceptions=True)
server = app.server
# Define the app layout with a data table and a distplot
app.layout = html.Div([
    html.H1('Welcome to The NFL W/L Dashboard by Spencer Prentiss',style={'padding-left':'100px','padding-bottom':'20px'}),

    dcc.Markdown(f"My Model's Current Win-Loss Record: {wins} - {loses} ( {wp}% )",
                 style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%', 'display': 'inline-block'}),
    dcc.Markdown(
        f"ESPN FPI Win-Loss Record: {espn_wins} - {espn_loses} ( {ewp}% )",
        style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%', 'display': 'inline-block'}),
    # dcc.Markdown(
    #     f"Vegas Win-Loss Record: {vegas_wins} - {vegas_loses} ( {vwp}%. )",
    #     style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%', 'display': 'inline-block'}),
    dash_table.DataTable(
        id='data-table',
        columns=[{'name': col, 'id': col} for col in table_data.columns],
        data=table_data.to_dict('records'),
        style_table={'textAlign': 'center'},  # Set table background color
        style_cell={'textAlign': 'center'},
        style_header={'textAlign': 'center'},
        row_selectable='single',  # Allow single-row selection
        selected_rows=[],
    ),

    dcc.Markdown(f"Select a matchup above (by clicking on one of the small circles on the left) to view stats about the game and the teams",
                 style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%', 'display': 'inline-block'}),

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Matchup Win Probability', value='tab-1'),
        dcc.Tab(label='Teams Performance Distribution', value='tab-2'),
        # dcc.Tab(label='In Game Performance Difference', value='tab-3'),
    ], style={  # Change the text color of the tabs
    }),

    html.Div(id='tabs-content'),

    html.Div(graph3, style={'font-weight': 'bold'}),
], style={'padding': '10px'},className="dbc dbc-row-selectable")


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(graph4, style={'width': '100%', 'display': 'inline-block'})
    elif tab == 'tab-2':
        return html.Div([html.Div(graph1, style={'width': '100%', 'display': 'inline-block'}),
                         dcc.Markdown(f"This chart is a distribution chart of each teams (in the selected matchup) performance. What does this "
                                      f"mean? Each team has a baseline performance score. For example Team A "
                                      f"has a score of 10%. This corresponds to them being better than the average "
                                      f"team by 10%. Each team also has a variance which tells us how variable their "
                                      f"performance is. The higher the number the greater variance in week to week "
                                      f"performance you can expect. What does this show us? Essentially you can look at "
                                      f"the chart an see how close teams are. The more they overlap the closer they are and vice versa. "
                                      f"",style={'padding': '10px'})],
                        style={'display': 'grid', 'grid-template-columns': '3fr 1fr','padding-bottom': '10px'})
    elif tab == 'tab-3':
        return html.Div(graph2, style={'width': '100%', 'display': 'inline-block'})


# Define a callback function to update the distplot based on the clicked row
@app.callback(
    Output('distplot', 'figure'),
    Input('data-table', 'selected_rows'),
    State('distplot', 'figure')
)
def update_distplot(selected_cell, current_fig):
    if not selected_cell:
        return current_fig

    # Extract the selected row's index
    selected_index = selected_cell[0]

    # Extract data from the selected row
    home_team_dvoa = df.iloc[selected_index]['Home_Team_DVOA']
    home_team_variance = df.iloc[selected_index]['Home_Team_Variance']
    away_team_dvoa = df.iloc[selected_index]['Away_Team_DVOA']
    away_team_variance = df.iloc[selected_index]['Away_Team_Variance']

    # Update the distplot data with the new mean and variance
    updated_x1 = np.random.normal(home_team_dvoa, np.sqrt(home_team_variance), 1000)
    updated_x2 = np.random.normal(away_team_dvoa, np.sqrt(away_team_variance), 1000)

    ht = df.iloc[selected_index]['Home_Team']
    at = df.iloc[selected_index]['Away_Team']

    ht_color = df.iloc[selected_index]['Home_Color']
    at_color = df.iloc[selected_index]['Away_Color']

    group_labels = [ht, at]

    colors = [ht_color, at_color]

    updated_fig = ff.create_distplot([updated_x1, updated_x2], group_labels, bin_size=0.5, curve_type='normal',
                                     colors=colors)
    updated_fig.update_layout(title_text=f'Teams Performance Distribution ( {ht} vs. {at} )')

    updated_fig.update_xaxes(title_text='Performance Above Average (%)')  # Add X-axis label
    updated_fig.update_yaxes(title_text='Probability')  # Add Y-axis label

    return updated_fig


# Define a callback function to update the distplot based on the clicked row
@app.callback(
    Output('diffplot', 'figure'),
    Input('data-table', 'selected_rows'),
    State('diffplot', 'figure')
)
def update_diffplot(selected_cell, current_fig):
    if not selected_cell:
        return current_fig

    # Extract the selected row's index
    selected_index = selected_cell[0]

    # Extract data from the selected row
    home_team_dvoa = df.iloc[selected_index]['Home_Team_DVOA'] / 100
    home_team_variance = df.iloc[selected_index]['Home_Team_Variance'] / 100
    away_team_dvoa = df.iloc[selected_index]['Away_Team_DVOA'] / 100
    away_team_variance = df.iloc[selected_index]['Away_Team_Variance'] / 100

    # Update the distplot data with the new mean and variance
    updated_x1 = np.random.normal(home_team_dvoa, np.sqrt(home_team_variance), 10000)
    updated_x2 = np.random.normal(away_team_dvoa, np.sqrt(away_team_variance), 10000)

    ht = df.iloc[selected_index]['Home_Team']
    at = df.iloc[selected_index]['Away_Team']

    ht_color = df.iloc[selected_index]['Home_Color']

    diff = updated_x1 - updated_x2

    # Create a histogram trace
    histogram_trace = go.Histogram(x=diff, histnorm='probability density',
                                   name='Difference in Performance Between Home & Away', marker=dict(color=ht_color))

    # Create a vertical line trace at X=0
    vertical_line_trace = go.Scatter(
        x=[0, 0],
        y=[0, 1],
        mode='lines',
        name='Vertical Line at X=0 ( More to the Right is Good for the Home Team )',
        line=dict(color='red')
    )

    updated_figure = {
        'data': [histogram_trace, vertical_line_trace],  # Update with the histogram and vertical line traces
        'layout': go.Layout(title=f'In Game Performance Difference ( {ht} vs. {at} )')
    }

    return updated_figure


# Define a callback function to update the distplot based on the clicked row
@app.callback(
    Output('pie-chart', 'figure'),
    Input('data-table', 'selected_rows'),
    State('pie-chart', 'figure')
)
def update_piechart(selected_cell, current_fig):
    if not selected_cell:
        return current_fig

    # Extract the selected row's index
    selected_index = selected_cell[0]

    # Extract data from the selected row

    home_team_wp = df.iloc[selected_index]['Home_Team_WP']
    away_team_wp = df.iloc[selected_index]['Away_Team_WP']

    ht = df.iloc[selected_index]['Home_Team']
    at = df.iloc[selected_index]['Away_Team']

    ht_color = df.iloc[selected_index]['Home_Color']
    at_color = df.iloc[selected_index]['Away_Color']

    # Create a histogram trace
    pie_chart = go.Pie(labels=[ht, at], values=[home_team_wp[:-1], away_team_wp[:-1]], pull=[0.15, 0],
                       marker=dict(colors=[ht_color, at_color], line=dict(color='#000000', width=2)),
                       hoverinfo='label+percent')

    updated_figure = {
        'data': [pie_chart],  # Update with the histogram and vertical line traces
        'layout': go.Layout(title=f'Win % ( {ht} vs. {at} )'),
    }

    return updated_figure


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
