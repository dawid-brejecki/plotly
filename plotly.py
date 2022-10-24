#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go


# In[2]:


# NUMPY

# konwertowanie do numpy array
mylist = [1,2.3,4]
arr = np.array(mylist)

# sprawdzanie typu
type(arr)

# tworzenie arraya przykladowego
a = np.arange(0,10)
# od zera do 11 co 2
a = np.arange(0,12,2)
# tabela zer 5x5
np.zeros((5,5))
# tabela jedynek
np.ones((2,4))

# losowa liczba z zakresu 0,100
np.random.randint(0,100)
# losowa tablica z zakresu 1,100 - wymiary 5x5
np.random.randint(0,100,(5,5))
# liczby ukladajace sie liniowo od 0 do 10, liczb ma byc 6
np.linspace(0,10,6)

# aby otrzymywac zawsze takie same liczby losowe, trzeba ustawic to samo ziarno i uruchamiac je tuz przed generacja wynikow
np.random.seed(101)
np.random.randint(0,100,10)

# max, min, srednia, pozycja max (indeks)
# INDEKSY W PYTHON ZACZYNAJA SIE OD 0
a.max()
a.min()
a.mean()
a.argmax()

# zmiana wymiaru tablicy
a.reshape(2,3)

# pobieranie pojedynczego elementu w 6 wierszu i 3 kolumnie
mat=np.arange(0,100).reshape(10,10)
mat[5,2]
# pobieranie całej kolumny trzeciej
mat[:,2]
# całego wiersza
mat[1,:]
# wartosci z tablicy wiekszych od 50
mat[mat>50]


# In[3]:


# PANDAS

df = pd.read_csv('/Users/user1/Desktop/salaries.csv')

# pobieranie tylko kolumny
# WAZNA JEST WIELKOSC LITER W PYTHONIE
df['Salary']

# pobieranie wiecej kolumn
df[['Salary', 'Age']]

# srednia wartosc w kolumnie
df['Salary'].mean()

# wiersze o Age > 30
df[df['Age']>30]

# unikalne wartosci Age
df['Age'].unique()

# liczba unikalnych wartosci Age
df['Age'].nunique()

# lista nazw kolumn
df.columns

# info
df.info()

# statystyczne podsumowanie
df.describe()

# z numpy do dataframe, sam nazywam kolumny

mat = np.arange(0,10).reshape(5,2)
df = pd.DataFrame(data=mat, columns = ['A', 'B'])


# In[4]:


# PLOTLY

# SCATTERPLOT
# stosowany dla dwoch zmiennych, sprawdzenie czy jest korelacja

np.random.seed(42)
random_x=np.random.randint(1,101,100)
random_y=np.random.randint(1,101,100)

data = [go.Scatter(x=random_x,
                   y=random_y,
                   mode='markers',
                   marker = dict(                   # wielkosc i kolor punktow
                       size= 12,
                       color='rgb(51,204,153)',
                       symbol='pentagon',
                       line={'width':2}             # kontur punktu
                   ))]
layout = go.Layout(title='hello', 
                  xaxis={'title':'MY X AXIS'},      # to czy uzyjemy {} czy dict() wszystko jedno
                  yaxis=dict(title='MY Y AXIS'),
                  hovermode='closest')              # w jaki sposob pojawiaja sie dane jak na nie najedziemy kursorem
fig = go.Figure(data=data, layout = layout)
pyo.plot(fig, filename='scatter.html')


# In[5]:


# LINEPLOT
# do ukazywania trendu w czasie

np.random.seed(56)
x_values = np.linspace(0,1,100)
y_values = np.random.randn(100)
trace0 = go.Scatter(x=x_values, y=y_values+5,
                  mode='markers', name='markers')
trace1 = go.Scatter(x=x_values, y=y_values, 
                    mode='lines', name='mylines')
trace2 = go.Scatter(x=x_values, y=y_values-5, 
                    mode='lines+markers', name='myfavourite')
data = [trace0, trace1, trace2]
layout = go.Layout(title='Line Charts')
fig=go.Figure(data=data, layout=layout)
pyo.plot(fig)


# In[68]:


# LINEPLOT W PANDASIE

df = pd.read_csv('/Users/user1/Desktop/nst-est2017-alldata.csv')
df2 = df[df['DIVISION']=='1']
df2.set_index('NAME', inplace=True)
list_of_pop_col = [col for col in df2.columns if col.startswith('POP')]
df2 = df2[list_of_pop_col]
data = [go.Scatter(x=df2.columns,
                   y=df2.loc[name],
                   mode='lines',
                   name=name) for name in df2.index]
pyo.plot(data)


# In[72]:


# INNE ZADANIE PANDAS

df = pd.read_csv('/Users/user1/Desktop/2010YumaAZ.csv')
days = ['TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY', 'MONDAY']
# dla kazdego dnia robimy scatterplota
data=[]
for day in days:
    trace=go.Scatter(x=df['LST_TIME'],
                          y=df[df['DAY']==day]['T_HR_AVG'],
                          mode='lines',
                          name=day)
    data.append(trace)
layout = go.Layout(title='Daily temp avgs')
fig=go.Figure(data=data, layout=layout)
pyo.plot(fig)


# In[6]:


# BARPLOT
# do prezentowania danych kategorycznych, najczesciej sumy wystapien

df = pd.read_csv('/Users/user1/Desktop/2018WinterOlympics.csv')

# prosty barplot
data = [go.Bar(x=df['NOC'], y = df['Total'])]
layout = go.Layout(title='Medals')
fig = go.Figure(data, layout)
pyo.plot(fig)

# nested barplot - kolumny obok siebie
trace1 = go.Bar(x=df['NOC'], 
                y=df['Gold'], 
                name='Gold',
                )
trace2 = go.Bar(x=df['NOC'], 
                y=df['Silver'], 
                name='Silver',
                )
trace3 = go.Bar(x=df['NOC'], 
                y=df['Bronze'], 
                name='Bronze',
                )
data = [trace1, trace2, trace3]
layout = go.Layout(title='Medals')
fig = go.Figure(data, layout)
pyo.plot(fig)

# zakumulowany barplot
trace1 = go.Bar(x=df['NOC'], 
                y=df['Gold'], 
                name='Gold',
                )
trace2 = go.Bar(x=df['NOC'], 
                y=df['Silver'], 
                name='Silver',
                )
trace3 = go.Bar(x=df['NOC'], 
                y=df['Bronze'], 
                name='Bronze',
                )
data = [trace1, trace2, trace3]
layout = go.Layout(title='Medals', barmode='stack')
fig = go.Figure(data, layout)
pyo.plot(fig)


# In[100]:


# Barplot odwrocony o 90 stopni
# zamieniamy x z y i dopisujemy orientation

df = pd.read_csv('/Users/user1/Desktop/mocksurvey.csv', index_col=0) # 1-sza kolumna indeksuje
data=[go.Bar(x=df[response], y=df.index, name=response, orientation = 'h') for response in df.columns]
layout = go.Layout(title='Results', barmode='stack')
fig = go.Figure(data, layout)
pyo.plot(fig)


# In[7]:


# BUBBLE PLOTS
# jak scatterplot, ale dodajemy trzecia zmienna, ktora odzwierciedla wielkosc markerow
df = pd.read_csv('/Users/user1/Desktop/mpg.csv')
df = df[df['horsepower'].apply(lambda x:x.isnumeric())]    # zmiana na int  
df['horsepower'] = df['horsepower'].astype('int64')        # zmiana na int
data=[go.Scatter(x=df['horsepower'],y= df['mpg'],
                text=df['name'], # tekst jaki chcemy wyswietlic przy najechaniu kursora
                mode='markers',
                marker=dict(size=df['weight']/100,
                           color=df['cylinders'],
                           showscale = True))]
layout = go.Layout(title='Results')
fig = go.Figure(data, layout)
pyo.plot(fig)


# In[119]:


# BOXPLOT
# wyswietla rozklad danych przez kwartyle
# ukazuje - kolejno - max, 75kwart, mediane, 25 kwart, min
# ukazuje tez outliery - obserwacje wieksze 1,5 raza od q3 i odwrotnie

snodgrass = [.209,.205,.196,.210,.202,.207,.224,.223,.220,.201]
twain = [.225,.262,.217,.240,.230,.229,.235,.217]

data = [
    go.Box(
        y=snodgrass,
        name='QCS', boxpoints = 'all', jitter=0.3, pointpos=2.0 # 2 po prawej, 1 wsrodku, 0 po lewo
    ),
    go.Box(
        y=twain,
        name='MT', boxpoints='outliers'
    )
]
layout = go.Layout(
    title = 'Comparison of three-letter-word frequencies<br>\
    between Quintus Curtius Snodgrass and Mark Twain'
)
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig, filename='box3.html')

# wybieranie 30 losowych danych z df['rings'], bez powtorzen
np.random.choice(df['rings'], 30, replace=False)


# In[8]:


# HISTOGRAM
# ukazuje rozklad zmiennej ciaglej w interwalach jakie okreslimy
df = pd.read_csv('/Users/user1/Desktop/abalone.csv')


# create a data variable:
data = [go.Histogram(
    x=df['length'],
    xbins=dict(start=0,end=1,size=.02), # zakres danych od 0 do 1 - wielkosc bina 0.02
)]

# add a layout
layout = go.Layout(
    title="Shell lengths from the Abalone dataset"
)

# create a fig from data & layout, and plot the fig
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig, filename='solution6.html')


# In[9]:


# DISTPLOT
# rozklad zmiennej ciaglej
import plotly.offline as pyo
import plotly.figure_factory as ff

snodgrass = [.209,.205,.196,.210,.202,.207,.224,.223,.220,.201]
twain = [.225,.262,.217,.240,.230,.229,.235,.217]

hist_data = [snodgrass,twain]
group_labels = ['Snodgrass','Twain']

fig = ff.create_distplot(hist_data, group_labels, bin_size=[.005,.005])
pyo.plot(fig, filename='SnodgrassTwainDistplot.html')


# In[10]:


# HEATMAP
# wizualizuje 3 zmienne - ciagla lub kategoryczna jako x i y oraz ciagla trzecia jako kolor

#######
# Heatmap of temperatures for Santa Barbara, California
######
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv('/Users/user1/Desktop/2010SantaBarbaraCA.csv')


data = [go.Heatmap(
    x=df['DAY'],
    y=df['LST_TIME'],
    z=df['T_HR_AVG'].values.tolist(),        # Z MUSI BYĆ LISTĄ!!! nie kolumna dataframe
    colorscale='Jet'                         # moga byc rozne skale kolorow
    
)]

layout = go.Layout(
    title='Hourly Temperatures, June 1-7, 2010 in<br>\
    Santa Barbara, CA USA'
)
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig, filename='Santa_Barbara.html')


# In[11]:


# multiple heatmaps - subploty

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly import subplots
import pandas as pd

df1 = pd.read_csv('/Users/user1/Desktop/2010SitkaAK.csv')
df2 = pd.read_csv('/Users/user1/Desktop/2010SantaBarbaraCA.csv')
df3 = pd.read_csv('/Users/user1/Desktop/2010YumaAZ.csv')


trace1 = go.Heatmap(
    x=df1['DAY'],
    y=df1['LST_TIME'],
    z=df1['T_HR_AVG'],
    colorscale='Jet',
    zmin = 5, zmax = 40 # warto dawac ten sam zakres danych w sobplotach
)
trace2 = go.Heatmap(
    x=df2['DAY'],
    y=df2['LST_TIME'],
    z=df2['T_HR_AVG'],
    colorscale='Jet',
    zmin = 5, zmax = 40
)
trace3 = go.Heatmap(
    x=df3['DAY'],
    y=df3['LST_TIME'],
    z=df3['T_HR_AVG'],
    colorscale='Jet',
    zmin = 5, zmax = 40
)

fig = subplots.make_subplots(rows=1, cols=3,
    subplot_titles=('Sitka, AK','Santa Barbara, CA', 'Yuma, AZ'),
    shared_yaxes = True,              # tylko jedna skala y
)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(                             # pozniejsza zmiana layoutu
    title='Hourly Temperatures, June 1-7, 2010'
)
pyo.plot(fig, filename='AllThree.html')


# In[12]:


import dash
from dash import dcc
from dash import html


# In[13]:


# prosty dash

app = dash.Dash()  # tworzenie aplikacji
app.layout = html.Div(children = [
    html.H1('Hello Dash!'),
    html.Div('Dash: Web Dashboards with Python'),
    dcc.Graph(id='example',
             figure={'data':[
                 {'x':[1,2,3], 'y': [4,1,2], 'type':'bar', 'name':'SF'},
                 {'x':[1,2,3], 'y': [2,4,5], 'type':'bar', 'name':'NYC'}],
                     'layout':{'title':'BAR PLOTS!'}})
])
                    
if __name__ == '__main__':
    app.run_server()


# In[18]:


# prosty dash 2

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(
        children='Dash: A web application framework for Python.',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                },
                'title': 'Dash Data Visualization'
            }
        }
    )],
    style={'backgroundColor': colors['background']}
)

if __name__ == '__main__':
    app.run_server()


# In[1]:


# wkladanie wykresow plotly do dash

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

app = dash.Dash()

np.random.seed(42)
random_x = np.random.randint(1,101,100)
random_y = np.random.randint(1,101,100)

app.layout = html.Div([dcc.Graph(id='scatterplot',
                    figure = {'data':[
                            go.Scatter(
                            x=random_x,
                            y=random_y,
                            mode='markers',
                            marker = {
                                'size':12,
                                'color': 'rgb(51,204,153)',
                                'symbol':'pentagon',
                                'line':{'width':2}
                            }
                            )],
                    'layout':go.Layout(title='My Scatterplot',
                                        xaxis = {'title':'Some X title'})}
                    ),
                    dcc.Graph(id='scatterplot2',
                                        figure = {'data':[
                                                go.Scatter(
                                                x=random_x,
                                                y=random_y,
                                                mode='markers',
                                                marker = {
                                                    'size':12,
                                                    'color': 'rgb(200,204,53)',
                                                    'symbol':'pentagon',
                                                    'line':{'width':2}
                                                }
                                                )],
                                        'layout':go.Layout(title='Second Plot',
                                                            xaxis = {'title':'Some X title'})}
                                        )])

if __name__ == '__main__':
    app.run_server()


# In[2]:


sudo lsof -i:8080


# In[23]:


# plotly - pandas - dash


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

# Launch the application:
app = dash.Dash()

# Create a DataFrame from the .csv file:
df = pd.read_csv('/Users/user1/Desktop/OldFaithful.csv')


# Create a Dash layout that contains a Graph component:
app.layout = html.Div([
    dcc.Graph(
        id='old_faithful',
        figure={
            'data': [
                go.Scatter(
                    x = df['X'],
                    y = df['Y'],
                    mode = 'markers'
                )
            ],
            'layout': go.Layout(
                title = 'Old Faithful Eruption Intervals v Durations',
                xaxis = {'title': 'Duration of eruption (minutes)'},
                yaxis = {'title': 'Interval to next eruption (minutes)'},
                hovermode='closest'
            )
        }
    )
])

# Add the server clause:
if __name__ == '__main__':
    app.run_server()


# In[24]:


# DASH HTML COMPONENTS

app = dash.Dash()

app.layout = html.Div([               # mamy dwa komponenty w jednym glownym
    'This is the outermost Div',
    html.Div(
        'This is an inner Div',
        style={'color':'blue', 'border':'2px blue solid', 'borderRadius':5,
        'padding':10, 'width':220}
    ),
    html.Div(
        'This is another inner Div',
        style={'color':'green', 'border':'2px green solid',
        'margin':10, 'width':220}
    ),
],
# this styles the outermost Div:
style={'width':500, 'height':200, 'color':'red', 'border':'2px red dotted'})

if __name__ == '__main__':
    app.run_server()


# In[25]:


# DASH CORE COMPONENTS
# TYLKO NIEKTORE TUTAJ SA UKAZANE
app = dash.Dash()

app.layout = html.Div([

    # DROPDOWN https://dash.plot.ly/dash-core-components/dropdown
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),

    html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF'],
        multi=True
    ),

    # SLIDER https://dash.plot.ly/dash-core-components/slider
    html.Label('Slider'),
    html.P(
    dcc.Slider(
        min=-5,
        max=10,
        step=0.5,
        marks={i: i for i in range(-5,11)},
        value=-3
    )),

    # RADIO ITEMS https://dash.plot.ly/dash-core-components/radioitems
    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    )
], style={'width': '50%'})

if __name__ == '__main__':
    app.run_server()


# In[27]:


# CALLBACKS - łączenie input i output

from dash.dependencies import Input, Output


# In[35]:


app = dash.Dash()
app.layout = html.Div([
    
    # wlozenie komponentu input, na razie bez polaczenia
    dcc.Input(id='my-id', value='Initial Text', type='text'), # to niejest ten input co z import input
   # tu bedzue output
    html.Div(id='my-div', style={'border':'2px blue solid'})
])

# polaczenie outputu z inputem
@app.callback(Output(component_id='my-div', component_property='children'), # children - domyslnie
             [Input(component_id='my_id', component_property='value')])


# definicja funkcji jest potrzebna by wskazac, co bedzie zwracac na ekranie
def update_output_div(input_value):
    return "You entered: {}".format(input_value)

if __name__ == '__main__':
    app.run_server()


# In[36]:


app = dash.Dash()

app.layout = html.Div([
    dcc.Input(id='my-id', value='initial value', type='text'),
    html.Div(id='my-div')
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'You\'ve entered "{}"'.format(input_value)

if __name__ == '__main__':
    app.run_server()


# In[47]:


# callback z wykresem

df = pd.read_csv('/Users/user1/Desktop/gapminderDataFiveYear.csv')


app = dash.Dash()


# https://dash.plot.ly/dash-core-components/dropdown

year_options = [] # zapelniamy opcje z listy rozwijanej
for year in df['year'].unique():
    year_options.append({'label':str(year),'value':year})

app.layout = html.Div([ # mamy tylko dwa komponenty - wykres i dropdown
    dcc.Graph(id='graph'),
    dcc.Dropdown(id='year-picker',options=year_options,value=df['year'].min())
])

# laczymy input z dropdown z output wykresem

@app.callback(Output('graph', 'figure'), # pominalem pisanie component... po prostu wartosci
              [Input('year-picker', 'value')])

# co ma pokazac funkcja na ekranie - nazwa argumentu moze byc dowolna - to po prostu input jaki dajemy
def update_figure(selected_year): # dwa filtrowania - jedno po roku
    filtered_df = df[df['year'] == selected_year]
    traces = []   # drugie po kontynencie
    for continent_name in filtered_df['continent'].unique():
        df_by_continent = filtered_df[filtered_df['continent'] == continent_name]
        traces.append(go.Scatter(
            x=df_by_continent['gdpPercap'],
            y=df_by_continent['lifeExp'],
            text=df_by_continent['country'],
            mode='markers',
            opacity=0.7,
            marker={'size': 15},
            name=continent_name
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            yaxis={'title': 'Life Expectancy'},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server()


# In[14]:


# 2 inputy, 1 output

app = dash.Dash()

df = pd.read_csv('/Users/user1/Desktop/mpg.csv')


features = df.columns

app.layout = html.Div([

        html.Div([
            dcc.Dropdown(
                id='xaxis',
                options=[{'label': i.title(), 'value': i} for i in features],
                value='displacement'
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis',
                options=[{'label': i.title(), 'value': i} for i in features],
                value='acceleration'
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    dcc.Graph(id='feature-graphic')
], style={'padding':10})

@app.callback(
    Output('feature-graphic', 'figure'),
    [Input('xaxis', 'value'),
     Input('yaxis', 'value')])
def update_graph(xaxis_name, yaxis_name):
    return {
        'data': [go.Scatter(
            x=df[xaxis_name],
            y=df[yaxis_name],
            text=df['name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={'title': xaxis_name.title()},
            yaxis={'title': yaxis_name.title()},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server()


# In[49]:


# 2 inputy, 2 outputy - ROZDZIELNE

app = dash.Dash()

df = pd.read_csv('/Users/user1/Desktop/wheels.csv')


app.layout = html.Div([
    dcc.RadioItems(
        id='wheels',
        options=[{'label': i, 'value': i} for i in df['wheels'].unique()],
        value=1
    ),
    html.Div(id='wheels-output'),

    html.Hr(),  # add a horizontal rule
    dcc.RadioItems(
        id='colors',
        options=[{'label': i, 'value': i} for i in df['color'].unique()],
        value='blue'
    ),
    html.Div(id='colors-output')
], style={'fontFamily':'helvetica', 'fontSize':18})

@app.callback(
    Output('wheels-output', 'children'),
    [Input('wheels', 'value')])
def callback_a(wheels_value):
    return 'You\'ve selected "{}"'.format(wheels_value)

@app.callback(
    Output('colors-output', 'children'),
    [Input('colors', 'value')])
def callback_b(colors_value):
    return 'You\'ve selected "{}"'.format(colors_value)

if __name__ == '__main__':
    app.run_server()


# In[52]:


############################# Important Point ##############################
#################################
###    Please add debug = True in line 399 if you are running the code using Anaconda Prompt 
###    and want to autoupdate the changes while making the changes.  

import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np


## reading the dataset 
pd_2 = pd.read_csv('/Users/user1/Desktop/retail_sales.csv')

pd_2['Date'] = pd.to_datetime(pd_2['Date'], format='%Y-%m-%d')


################################ total sales Month level  ###################################
monthly_sales_df = pd_2.groupby(['month','Month']).agg({'Weekly_Sales':'sum'}).reset_index()


################################ holiday sales month lvl #####################################
holiday_sales = pd_2[pd_2['IsHoliday'] == 1].groupby(['month'])['Weekly_Sales'].sum().reset_index().rename(columns={'Weekly_Sales':'Holiday_Sales'})

############################# combined #########################
monthly_sales_df  = pd.merge(holiday_sales,monthly_sales_df,on = 'month', how = 'right').fillna(0)
 
############################## rounding sales to 1 decimal #############################
monthly_sales_df['Weekly_Sales'] = monthly_sales_df['Weekly_Sales'].round(1)
monthly_sales_df['Holiday_Sales'] = monthly_sales_df['Holiday_Sales'].round(1)


###################### weekly sales #########################
weekly_sale = pd_2.groupby(['month','Month','Date']).agg({'Weekly_Sales':'sum'}).reset_index()
weekly_sale['week_no'] = weekly_sale.groupby(['Month'])['Date'].rank(method='min')


########################### store level sales #######################
store_df=pd_2.groupby(['month','Month','Store']).agg({'Weekly_Sales':'sum'}).reset_index()
store_df['Store'] = store_df['Store'].apply(lambda x: 'Store'+" "+str(x))
store_df['Weekly_Sales'] = store_df['Weekly_Sales'].round(1)


######################## dept level sales #########################
dept_df=pd_2.groupby(['month','Month','Dept']).agg({'Weekly_Sales':'sum'}).reset_index()
dept_df['Dept'] = dept_df['Dept'].apply(lambda x: 'Dept'+" "+str(x))
dept_df['Weekly_Sales'] = dept_df['Weekly_Sales'].round(1)

#########################################


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"


navbar = dbc.Navbar( id = 'navbar', children = [
    dbc.Row([
        dbc.Col(html.Img(src = PLOTLY_LOGO, height = "70px")),
        dbc.Col(
            dbc.NavbarBrand("Retail Sales Dashboard", style = {'color':'white', 'fontSize':'25px','fontFamily':'Times New Roman'}
                            )
            
            )
        
        
        ],align = "center",
        # no_gutters = True
        ),
    
    
    ], color = '#090059')


card_content_dropdwn = [
    dbc.CardBody(
        [
            html.H6('Select Months', style = {'textAlign':'center'}),
            
            dbc.Row([
                
                dbc.Col([
                    
                    html.H6('Current Period'),
                    
                    dcc.Dropdown( id = 'dropdown_base',
        options = [
            {'label':i, 'value':i } for i in monthly_sales_df.sort_values('month')['Month']
        
            ],
        value = 'Feb',
        
        )
                    
                    
                    ]),
                
                dbc.Col([
                    
                    html.H6('Reference Period'),
                    
                    dcc.Dropdown( id = 'dropdown_comp',
        options = [
            {'label':i, 'value':i } for i in monthly_sales_df.sort_values('month')['Month']
        
            ],
        value = 'Jan',
        
        )
                    
                    
                    ]),
                
                
                
                
                ])
            
            ]
        )
    
    
    
    ]


body_app = dbc.Container([
    
    html.Br(),
    html.Br(),
    
    dbc.Row([
        dbc.Col([dbc.Card(card_content_dropdwn,style={'height':'150px'})],width = 4),
        dbc.Col([dbc.Card(id = 'card_num1',style={'height':'150px'})]),
        dbc.Col([dbc.Card(id = 'card_num2',style={'height':'150px'})]),
        dbc.Col([dbc.Card(id = 'card_num3',style={'height':'150px'})]),

        ]),
    
    html.Br(),
    html.Br(),
    
    dbc.Row([
        dbc.Col([dbc.Card(id = 'card_num4',style={'height':'350px'})]),
        dbc.Col([dbc.Card(id = 'card_num5',style={'height':'350px'})]),
        dbc.Col([dbc.Card(id = 'card_num6',style={'height':'350px'})]),

        ]),
    
    html.Br(),
    html.Br()
    
    
    ], 
    style = {'backgroundColor':'#f7f7f7'},
    fluid = True)


app.layout = html.Div(id = 'parent', children = [navbar,body_app])


@app.callback([Output('card_num1', 'children'),
               Output('card_num2', 'children'),
               Output('card_num3', 'children'),
               Output('card_num4', 'children'),
               Output('card_num5', 'children'),
               Output('card_num6', 'children'),
               ],
              [Input('dropdown_base','value'), 
                Input('dropdown_comp','value')])
def update_cards(base, comparison):
    
    sales_base = monthly_sales_df.loc[monthly_sales_df['Month']==base].reset_index()['Weekly_Sales'][0]
    sales_comp = monthly_sales_df.loc[monthly_sales_df['Month']==comparison].reset_index()['Weekly_Sales'][0]

    diff_1 = np.round(sales_base -sales_comp,1)
    
    holi_base = monthly_sales_df.loc[monthly_sales_df['Month']==base].reset_index()['Holiday_Sales'][0]
    holi_comp = monthly_sales_df.loc[monthly_sales_df['Month']==comparison].reset_index()['Holiday_Sales'][0]

    diff_holi = np.round(holi_base -holi_comp,1)
    
    
    base_st_ct = pd_2.loc[pd_2['Month']==base,'Store'].drop_duplicates().count()
    comp_st_ct = pd_2.loc[pd_2['Month']==comparison,'Store'].drop_duplicates().count()

    diff_store = np.round(base_st_ct-comp_st_ct,1)
    
    
    weekly_base = weekly_sale.loc[weekly_sale['Month']==base].reset_index()
    weekly_comp = weekly_sale.loc[weekly_sale['Month']==comparison].reset_index()
    
    
    
    store_base = store_df.loc[store_df['Month']==base].sort_values('Weekly_Sales',ascending = False).reset_index()[:10]
    store_comp = store_df.loc[store_df['Month']==comparison].sort_values('Weekly_Sales',ascending = False).reset_index()[:10]
    
    dept_base = dept_df.loc[dept_df['Month']==base].sort_values('Weekly_Sales',ascending = False).reset_index()[:10]
    dept_base=dept_base.rename(columns = {'Weekly_Sales':'Weekly_Sales_base'})
    dept_comp = dept_df.loc[dept_df['Month']==comparison].sort_values('Weekly_Sales',ascending = False).reset_index()
    dept_comp=dept_comp.rename(columns = {'Weekly_Sales':'Weekly_Sales_comp'})
    
    merged_df=pd.merge(dept_base, dept_comp, on = 'Dept', how = 'left')
    merged_df['diff'] = merged_df['Weekly_Sales_base']-merged_df['Weekly_Sales_comp']


    
    fig = go.Figure(data = [go.Scatter(x = weekly_base['week_no'], y = weekly_base['Weekly_Sales'],                                   line = dict(color = 'firebrick', width = 4),name = '{}'.format(base)),
                        go.Scatter(x = weekly_comp['week_no'], y = weekly_comp['Weekly_Sales'],\
                                   line = dict(color = '#090059', width = 4),name = '{}'.format(comparison))])

    
    fig.update_layout(plot_bgcolor = 'white',
                      margin=dict(l = 40, r = 5, t = 60, b = 40),
                      yaxis_tickprefix = '$',
                      yaxis_ticksuffix = 'M')


    fig2 = go.Figure([go.Bar(x = store_base['Weekly_Sales'], y = store_base['Store'], marker_color = 'indianred',name = '{}'.format(base),                             text = store_base['Weekly_Sales'], orientation = 'h',
                             textposition = 'outside'
                             ),
                 ])
        
        
    fig3 = go.Figure([go.Bar(x = store_comp['Weekly_Sales'], y = store_comp['Store'], marker_color = '#4863A0',name = '{}'.format(comparison),                             text = store_comp['Weekly_Sales'], orientation = 'h',
                             textposition = 'outside'
                             ),
                 ])
        
    fig2.update_layout(plot_bgcolor = 'white',
                       xaxis = dict(range = [0,'{}'.format(store_base['Weekly_Sales'].max()+3)]),
                      margin=dict(l = 40, r = 5, t = 60, b = 40),
                      xaxis_tickprefix = '$',
                      xaxis_ticksuffix = 'M',
                      title = '{}'.format(base),
                      title_x = 0.5)
    
    fig3.update_layout(plot_bgcolor = 'white',
                       xaxis = dict(range = [0,'{}'.format(store_comp['Weekly_Sales'].max()+3)]),
                      margin=dict(l = 40, r = 5, t = 60, b = 40),
                      xaxis_tickprefix = '$',
                      xaxis_ticksuffix = 'M',
                      title = '{}'.format(comparison),
                      title_x = 0.5)

    fig4 = go.Figure([go.Bar(x = merged_df['diff'], y = merged_df['Dept'], marker_color = '#4863A0',                              orientation = 'h',
                             textposition = 'outside'
                             ),
                 ])
        
    fig4.update_layout(plot_bgcolor = 'white',
                       margin=dict(l = 40, r = 5, t = 60, b = 40),
                      xaxis_tickprefix = '$',
                      xaxis_ticksuffix = 'M'
                     )


    
    if diff_1 >= 0:
        a =   dcc.Markdown( dangerously_allow_html = True,
                   children = ["<sub>+{0}{1}{2}</sub>".format('$',diff_1,'M')], style = {'textAlign':'center'})
        
    elif diff_1 < 0:
        
        a =    dcc.Markdown( dangerously_allow_html = True,
                   children = ["<sub>-{0}{1}{2}</sub>".format('$',np.abs(diff_1),'M')], style = {'textAlign':'center'})
            
    if diff_holi >= 0:
        b =   dcc.Markdown( dangerously_allow_html = True,
                   children = ["<sub>+{0}{1}{2}</sub>".format('$',diff_holi,'M')], style = {'textAlign':'center'})
        
    elif diff_holi < 0:
        
        b =   dcc.Markdown( dangerously_allow_html = True,
                   children = ["<sub>-{0}{1}{2}</sub>".format('$',np.abs(diff_holi),'M')], style = {'textAlign':'center'})
        
    if diff_store >= 0:
        c =   dcc.Markdown( dangerously_allow_html = True,
                   children = ["<sub>+{0}</sub>".format(diff_store)], style = {'textAlign':'center'})
        
    elif diff_store < 0:
        
        c =   dcc.Markdown( dangerously_allow_html = True,
                   children = ["<sub>-{0}</sub>".format(np.abs(diff_store))], style = {'textAlign':'center'})
        
        
    
    card_content = [
        
        dbc.CardBody(
            [
                html.H6('Total sales', style = {'fontWeight':'lighter', 'textAlign':'center'}),
                
                html.H3('{0}{1}{2}'.format("$", sales_base, "M"), style = {'color':'#090059','textAlign':'center'}),
                
                a
                
                ]
                   
            )  
        ]
    
    card_content1 = [
        
        dbc.CardBody(
            [
                html.H6('Holiday Sales', style = {'fontWeight':'lighter', 'textAlign':'center'}),
                
                html.H3('{0}{1}{2}'.format("$", holi_base, "M"), style = {'color':'#090059','textAlign':'center'}),
                
                b
                
                ]
                   
            )  
        ]
    
    card_content2 = [
        
        dbc.CardBody(
            [
                html.H6('Total Stores', style = {'fontWeight':'lighter', 'textAlign':'center'}),
                
                html.H3('{0}'.format( base_st_ct), style = {'color':'#090059','textAlign':'center'}),
                
                c
                
                ]
                   
            )  
        ]
    
    card_content3 = [
        
        dbc.CardBody(
            [
                html.H6('Weekly Sales Comparison', style = {'fontWeight':'bold', 'textAlign':'center'}),
                
                dcc.Graph(figure = fig, style = {'height':'250px'})
                
                
                ]
                   
            )  
        ]
    
    
    card_content4 = [
        
        dbc.CardBody(
            [
                html.H6('Stores with highest Sales', style = {'fontWeight':'bold', 'textAlign':'center'}),
                
                dbc.Row([
                    dbc.Col([dcc.Graph(figure = fig2, style = {'height':'300px'}),
                ]),
                    dbc.Col([dcc.Graph(figure = fig3, style = {'height':'300px'}),
                ])
                    
                    ])
                
                
                
                ]
                   
            )  
        ]
    
    card_content5 = [
        
        dbc.CardBody(
            [
                html.H6('Sales difference between Top departments ({} - {})'.format(base, comparison), style = {'fontWeight':'bold', 'textAlign':'center'}),
                
                dcc.Graph(figure = fig4, style = {'height':'300px'})
                
                
                ]
                   
            )  
        ]
    

    
    return card_content, card_content1, card_content2,card_content3,card_content4,card_content5


if __name__ == "__main__":
    app.run_server()
    #debug = True










# In[51]:


conda install -c conda-forge dash-bootstrap-components


# In[ ]:




