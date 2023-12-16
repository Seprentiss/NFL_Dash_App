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

wins = 139
loses = 71

espn_wins = 135
espn_loses = 75

# vegas_wins = 57
# vegas_loses = 30

wp = round(wins / (wins + loses) * 100, 2)
ewp = round(espn_wins / (espn_wins + espn_loses) * 100, 2)
# vwp = round(vegas_wins / (vegas_wins + vegas_loses) * 100, 2)

mean_x1 = 10  # Default mean for x1
variance_x1 = 5  # Default variance for x1
mean_x2 = 5  # Default mean for x2
variance_x2 = 10  # Default variance for x2

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

if np.mean(diff) < 0:
    hw_hl = f" The Home Team Loses by {round(np.mean(diff) * 2) / 2}"
else:
    hw_hl = f" The Home Team Wins by {round(np.mean(diff) * 2) / 2}"

histogram_trace = go.Histogram(x=diff, histnorm='probability density',
                               name='Home Team Point Spread', marker=dict(color="slategrey"))

updated_figure = {
    'data': [histogram_trace],  # Update with the histogram and vertical line traces
    'layout': go.Layout(title=f'Predicted Home Team Point Spread ( Home Team vs. Away Team ){hw_hl}',
                        xaxis=dict(title='Spread', range=[-60, 60]))
}

# Read data from a local CSV file
df = pd.read_csv('wp_data.csv', index_col=False)

pow_data = pd.read_csv('Pow_data.csv', index_col=False)

table_data = df.drop(
    columns=['Home_Team_DVOA', 'Home_Team_Variance', 'Home_Color', 'Home_Team_WP', 'Away_Team_DVOA',
             'Away_Team_Variance',
             "Away_Color", 'Away_Team_WP',"Game_Quality"])

graph1 = dcc.Graph(id='distplot', figure=fig),
graph2 = dcc.Graph(
    id='diffplot',
    figure={
        'data': [histogram_trace],
        'layout': go.Layout(
            title='Home Team Point Differential (Point Spread) (Now Showing Example Plot)',
            xaxis=dict(title='Spread',range=[-10, 10]),
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
    # html.H1('NFL W/L Dashboard by Spencer Prentiss',style={'padding-left':'100px','padding-bottom':'20px'}),
    #
    # html.H3('Welcome to The NFL W/L Dashboard',style={'padding-bottom':'20px','textAlign':'center'}),

    html.Header(
        html.H1("Spencer Prentiss' NFL Picks Dashboard", style={'color': '#fff','textAlign':'center'})
    ),

    html.Section(
        children=[
            html.P(
                "Welcome to my NFL game prediction model! I decided to make this model because of my love for "
                "football as well as me being competitive and wanting to try and compete with the best models out "
                "there. This was a lot of fun to work on and I hope to add to it as it has been a great exercise in Data "
                "Science and Machine learning."),
        ], style={'max-width': '1000px', 'margin': '20px auto', 'padding': '20px', 'background-color': '#2e4059',
               'border-radius': '8px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'}),

    html.Section(
        children=[

            html.P("How does this model work?", style={'font-weight': 'bold'}),

            html.P('''It generates a performance distribution for each team to predict potential weekly performance. 
            Then it compares two teams’ distributions to estimate the likelihood of one winning, what the margin of 
            victory will be, and how many total points will be scored in the matchup.'''),

            html.P('''Additional factors considered that could effect team performance'''),

            dcc.Markdown("""
                    - 🏟️ **Home Field Advantage:** Employs a weighted moving average model to consider the impact of home-field advantage throughout the season.
                    - 🏈 **Quarterback Performance:** Utilizes a formula to assess quarterback performance, factoring in adjustments for injuries and benching.
                    - 📅 **Bye Weeks Analysis:** Considers historical performance after bye weeks to accommodate teams returning from a bye and their potential impact on the game.
                """),

            html.P(
                    "Curious about my model's performance? Compare it with ESPN FPI, a leading model, right here on "
                    "this site. You'll find my predictions for each game, additional insights into team performance, "
                    "and a comprehensive power ranking for all teams."),

            # html.P(["For a deeper dive into the methodology and technical intricacies behind this project, "
            #         "explore the detailed write-up ",
            #         html.A("here", href="https://docs.google.com/document/d/1S3AXh6LxYXtjvHctYGUz_qzYhbfpmKyaWEFdrsZQ74s/edit?usp=sharing",target="_blank"),"."]),
            # html.Ul([
            #         html.Li("Currently, I’m working on adding approaches for point spread and point totals."),
            #         # Add more list items as needed
            #     ])

        ],
        style={'max-width': '1000px', 'margin': '20px auto', 'padding': '20px', 'background-color': '#2e4059',
               'border-radius': '8px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'}
    ),

    html.Section(
        children=[html.P("Enjoy your exploration of this project, and hopefully, you can glean some new insights from this."),

                html.P(["To check out more of my work you can head over to my ",
                    html.A("portfolio", href="https://seprentiss.github.io/portfolio/",target="_blank")," or my ", html.A("Linkedin", href="https://www.linkedin.com/in/spencerprentiss/",target="_blank"),"."]),

                html.P("Let the predictions commence, and let's see how well I can do!"),],

        style={'max-width': '1000px', 'margin': '20px auto', 'padding': '20px', 'background-color': '#2e4059',
               'border-radius': '8px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'}
        ),

    html.Section(
        children=[
            dcc.Markdown(f"My Model's Current Win-Loss Record: {wins} - {loses} ( {wp}% )",
                         style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%',
                                'display': 'inline-block'}),
            dcc.Markdown(
                f"ESPN FPI Win-Loss Record: {espn_wins} - {espn_loses} ( {ewp}% )",
                style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%', 'display': 'inline-block'}),
        ],
        style={'max-width': '800px', 'margin': '20px auto', 'padding': '20px', 'background-color': '#2e4059',
               'border-radius': '8px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'}
    ),
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

    html.Section(
        children=[
            dcc.Markdown(f"Select a matchup above (by clicking on one of the small circles on the left in the table) "
                         f"to view the predictions for each matchup. Click on the different tabs below to look at "
                         f"different metrics for the game.",
                         style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%',
                                'display': 'inline-block'}),
        ],
        style={'max-width': '1000px', 'margin': '20px auto', 'padding': '20px', 'background-color': '#2e4059',
               'border-radius': '8px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'}
    ),

    # dcc.Markdown(f"",
    #              style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%', 'display': 'inline-block'}),

    dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    id="tabs",
                    active_tab='tab-1',
                    children=[
                        dbc.Tab(
                            label='Matchup Win Probability',
                            tab_id='tab-1',
                            active_label_style={"background-color": "#2e4059"},
                            label_style={"background-color": "#595959","margin-right": "2px"},
                        ),
                        dbc.Tab(
                            label='Teams Performance Distribution',
                            tab_id='tab-2',
                            active_label_style={"background-color": "#2e4059"},
                            label_style={"background-color": "#595959","margin-right": "2px"}

                        ),
                        dbc.Tab(
                            label='Home Team Point Differential (Work In Progress)',
                            tab_id='tab-3',
                            active_label_style={"background-color": "#2e4059"},
                            label_style={"background-color": "#595959"}
                        ),
                    ],
                ),
            style={"background-color": "#222222"}),
            dbc.CardBody(html.Div(id='tabs-content')),
        ],style={"background-color": "white"}),


    # html.Div(id='tabs-content'),

    html.Section(
        children=[
            dcc.Markdown(f"The chart below shows the Power Rankings for the league and a 90% confidence interval "
                         f"for each teams average performance strength.",
                         style={'font-size': '20px', 'font-weight': 'bold', 'width': '100%',
                                'display': 'inline-block'}),
        ],
        style={'max-width': '1000px', 'margin': '20px auto', 'padding': '20px', 'background-color': '#2e4059',
               'border-radius': '8px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'}
    ),

    html.Div(graph3, style={'font-weight': 'bold'}),
], style={'padding': '10px'},className="dbc dbc-row-selectable")


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'active_tab'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(graph4, style={'width': '100%', 'display': 'inline-block'})
    elif tab == 'tab-2':
        return html.Div([html.Div(graph1, style={'width': '100%', 'display': 'inline-block'}),
                         dcc.Markdown(f"This chart serves as a theoretical representation of each teams "
                                      f"performance distribution in a selected matchup. Each team has a mean baseline "
                                      f"performance value aligned with the center of their performance curve. The curve's "
                                      f"width indicates the degree of week-to-week performance fluctuations (higher "
                                      f"variance results in wider curves). This visual tool allows for an intuitive "
                                      f"assessment of how close a matchup is; overlapping curves suggest a close game, "
                                      f"while distinct curves indicate one team is much better than the other.",
                                      style={'padding': '10px','background-color':'#595959'})],
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

    np.random.seed(42)
    if not selected_cell:
        return current_fig

    # Extract the selected row's index
    selected_index = selected_cell[0]

    # Extract data from the selected row
    home_team_dvoa = df.iloc[selected_index]['Home_Team_DVOA'] / 100 + 0.09532660918445379
    home_team_variance = df.iloc[selected_index]['Home_Team_Variance'] / 100
    away_team_dvoa = df.iloc[selected_index]['Away_Team_DVOA'] / 100
    away_team_variance = df.iloc[selected_index]['Away_Team_Variance'] / 100

    # Update the diffplot data with the new mean and variance
    updated_x1 = np.random.normal(home_team_dvoa, np.sqrt(home_team_variance), 10_000)
    updated_x2 = np.random.normal(away_team_dvoa, np.sqrt(away_team_variance), 10_000)

    ht = df.iloc[selected_index]['Home_Team']
    at = df.iloc[selected_index]['Away_Team']

    ht_color = df.iloc[selected_index]['Home_Color']
    avg_points = 21.77
    for i in range(len(updated_x1)):
        if updated_x1[i] < 0:
            updated_x1[i] = round(avg_points * (1 + updated_x1[i]))
        else:
            updated_x1[i] = round(avg_points * (1 + updated_x1[i]))

    for i in range(len(updated_x2)):
        if updated_x2[i] < 0:
            updated_x2[i] = round(avg_points * (1 + updated_x2[i]))
        else:
            updated_x2[i] = round(avg_points * (1 + updated_x2[i]))

    diff = updated_x1 - updated_x2

    margins = pd.read_csv("NFL_Margins.csv")

    data = np.array([])
    for i in range(len(margins)):
        tot = margins["MARGIN"].iloc[i]
        freq = margins["FREQ"].iloc[i]
        for j in range(freq):
            data = np.append(data, tot)

    # Parameters for the second normal distribution
    mu1, std1 = np.mean(diff), np.std(diff)

    # Number of samples
    num_samples = 10_000

    # Generate samples from the first and second distributions
    dist1_samples = np.random.choice(data, size=int(num_samples * 0.2), replace=True)
    dist2_samples = np.random.normal(mu1, std1, int(num_samples * 0.8))

    dist2_samples = [round(element) for element in dist2_samples]


    # Assign weights to the distributions
    weight_dist1 = 0.2
    weight_dist2 = 0.8

    # Combine the samples based on weights
    weighted_samples = np.concatenate([
        np.random.choice(dist1_samples, size=int(num_samples * weight_dist1)),
        np.random.choice(dist2_samples, size=int(num_samples * weight_dist2))
    ])

    # Create a histogram trace
    histogram_trace = go.Histogram(x=weighted_samples, histnorm='probability density',
                                   name='Home Team Point Spread', marker=dict(color=ht_color))


    mean = np.mean(weighted_samples)

    if mean < 0:
        hw_hl = f" The {ht} Lose by {round(mean*2)/2}"
    else:
        hw_hl = f" The {ht} Win by {round(mean*2)/2}"

    updated_figure = {
        'data': [histogram_trace],  # Update with the histogram and vertical line traces
        'layout': go.Layout(title=f'Predicted Home Team Point Spread ( {ht} vs. {at} ){hw_hl}',
                            xaxis=dict(title='Spread', range=[-60,60],tickvals = list(range(-60, 61, 5))))
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
    pie_chart = go.Pie(labels=[ht, at], values=[f"{home_team_wp[:-1]}", f"{away_team_wp[:-1]}"], pull=[0.15, 0],
                       marker=dict(colors=[ht_color, at_color], line=dict(color='#000000', width=2)),
                       hoverinfo='label+percent')

    updated_figure = {
        'data': [pie_chart],  # Update with the histogram and vertical line traces
        'layout': go.Layout(title=f'Win % ( {ht} vs. {at} )'),
    }

    return updated_figure

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
