
from matplotlib import cm
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bokeh.io import curdoc, show
from bokeh.layouts import row, column, layout, gridplot
from bokeh.models.callbacks import CustomJS
from bokeh.models import Legend, LegendItem
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, TapTool, OpenURL
from bokeh.models import Circle, Div, Paragraph, PreText
from bokeh.models import CDSView, IndexFilter, BoxAnnotation

from bokeh.models.widgets import Slider, TextInput, MultiSelect, Select, Button, AutocompleteInput
from bokeh.models.widgets import CheckboxGroup, RadioButtonGroup, RadioGroup

from bokeh.plotting import figure, output_file
from bokeh.transform import factor_mark, factor_cmap
from bokeh.colors import RGB
from bokeh import __version__

from scipy.special import softmax
from matplotlib.colors import Normalize

from bokeh.models import LinearColorMapper, ColorBar, FixedTicker, CustomJSTickFormatter

#from encodings import circularEnc, linearEnc
# Definition of useful functions
import numpy as np
import pandas as pd

# Definimos un encoding circular adecuado
def circularEnc(list_vals, keyString):
    """
    DESCRIPTION
    Produces a dataframe E, containing N=len(list_vals) positions equally distributed
    in a circle. 
    The N positions are associated to the N classes of a given category, respectively. 
    It is assumed that:
            - The classes in list_values are exhaustive (the encoding includes all the possible classes in the category)
            - The classes are defined by an integer that can be sorted

    INPUTS
        list_vals: list with values for all the categories
        keyString: String identifier for the encoding (ej. "weekday")

    OUTPUTS
        E: Encoding (dataframe with the class number and the position)
    """

    list_vals.sort()
    N=len(list_vals)
    x = np.arange(N)
    pos = np.vstack((np.cos(2 * np.pi * x / N), np.sin(2 * np.pi * x / N))).T
  
    E = pd.DataFrame(pos, list_vals, columns=['posx', 'posy'])
    E[keyString] = list_vals
    return E


# Definimos un encoding lineal  adecuado para el valor de una columna del dataframe de datos.
# Esta función debe ser global para poder usarla cuando cambiemos las cajas de texto con las 
# variables utilizadas.
# Recibe el dataframe (df) el nombre de la columna a usar (col) y el tipo ('ver'=vertical),
# por defecto horizontal
def linearEnc(df, col, tipo='ver'):
    from sklearn import preprocessing
    
    mms = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    if tipo == 'ver':
        pos = np.hstack((0*df[col].values[:, None],
                         mms.fit_transform(df[col].values[:, None])))
    else:
        pos = np.hstack(
            (mms.fit_transform(df[col].values[:, None]), 0*df[col].values[:, None]))
    return pos

# Definimos una función auxiliar para devolver un mapa de colores para una variable 
# determinada.
def getColorMap(df, variable):
 
    vals=df[variable].values
    if len(df[variable].unique())>20:
        [min,max]=df[variable].quantile([0.05,0.85])
        min=np.min(vals[np.nonzero(vals)])
    else:
        max=np.max(vals)
        min=np.min(vals)
        
    # META: Buscar un mapa de color adecuado, seleccionarlo automáticamente o 
    # como parámetro.
    if(len(df[variable].unique()) <= 20):
        # Suppose categorical
        print("Supuesto colormap categórico")
        colores = [RGB(*cm.turbo(i, bytes=True)[:4]) for i in Normalize(
                   vmin=min, vmax=max) (df[variable].values)]
        palette = "Turbo256"
    else:
        print("Supuesto colormap continuo")
        colores = [RGB(*cm.viridis(i, bytes=True)[:4]) for i in Normalize(
                vmin=min, vmax=max) (df[variable].values)]
        palette = "Viridis256"
        # print("Supuesto colormap continuo")
        # colores = [RGB(*cm.turbo(i, bytes=True)[:4]) for i in Normalize(
        #         vmin=min, vmax=max) (df[variable].values)]
        # palette = "Turbo256"
    
    color_mapper = LinearColorMapper(palette = palette,
                                 low = df[variable].min(), 
                                 high = df[variable].max())

    return colores, color_mapper



#######################################################
# VAMOS CON UNAS FUNCIONES PARA CALCULAR DISTANCIAS
# ENTRE LOS GRUPOS SELECCIONADOS POR EL USUARIO
#######################################################

# CalculaRegresionLogistica(s1,s2)
# Calcula el hiperplano que mejor separa los grupos
# de datos de los conjuntos de índices s1 y s2
# mediante regresión logística, y devuelve el
# vector director procesado para usar como criterio
# de ordenación.
def CalculaRegresionLogistica(s1, s2, lista):
    import sklearn as sk
    from sklearn.linear_model import LogisticRegression

    x = df.iloc[s1, :]
    if (pd.__version__ < '2.0'):
        x = x.append(df.iloc[s2, :])
    else:
        x = pd.concat([x, df.iloc[s2, :]], ignore_index=True)
    
    x = x[x.columns.intersection(lista)]
    y = pd.DataFrame([0]*len(s1) + [1]*len(s2), columns=list('T'))
    print(f'.      numpy: {np.__version__}')
    print(f'scikit-learn: {sk.__version__}')
    print('Realizando regresión...')
    LR = LogisticRegression(
        random_state=0, solver='liblinear', multi_class='ovr').fit(x, y.iloc[:, 0])
    print('Listo.')
    d = LR.coef_
    return d


# CalculaVectorOrdenado: recibe dos vectores con índices de elementos
# correspondientes a dos grupos seleccionados y una lista con nombres 
# de columnas y realiza las siguientes acciones:
#   1/ Realiza una regresión logística y devuelve el vector director
#       del hiperplano que separa ambos grupos.
#   3/ Ordena la lista de variables utilizada de mayor a menor por
#       el módulo de su componente asociada del vector director
#       como aproximación a aquellos que tienen más influencia en que ambos
#       grupos estén separados
# Devuelve un dataframe con la lista de variables y las distancias ordenado

def CalculaVectorOrdenado(s1, s2, lista_cols):
    # Tomamos los datos seleccionados de
    # ambos patches del data frame
    print("Calculando distancia entre grupos...")
    
    d = CalculaRegresionLogistica(s1, s2, lista_cols)
    
    print("Normalizando...")
    d = np.abs(d)
    # Normalizamos para que el máximo valga 1 y el resto como fracción de él.
    d = d/np.max(np.max(d))

    print("Creando nuevo dataframe...")
    # Y creamos un dataframe con ellas y las columnas del dataframe
    # global df, que contienen la lista de variables
    tdf = pd.DataFrame(d.T, index=df.columns.intersection(
         lista_cols).T, columns=list(['Distance']))
    # Ordenamos el dataframe por 'Distance'
    tdf.sort_values(by='Distance', inplace=True, ascending=False)

    return tdf



##########################################################
# Leemos y procesamos los datos de configuración
##########################################################

import json
f = open('mpconfig.json', encoding='utf-8')
conf = json.load(f)
f.close()

print("Leyendo archivo de configuración...")
print(conf["comment"]["text"])
print("Versión:")
print(conf["comment"]["version"])

# Datos
datapath = conf["data"]["datapath"]
filename = conf["data"]["filename"]

# Orden y coloreado
sortbyColumn =  conf["defaults"]["sortbyColumn"]
colorbyColumn = conf["defaults"]["colorbyColumn"]

# Cajas de selección
vbleEncodingHor = conf["defaults"]["box1"]
vbleEncodingVer = conf["defaults"]["box2"]
autocomplete = (conf["defaults"]["autocomplete"] != 'No')

# Columnas a usar, si vacía usamos todas
columnas = conf["columns"]["columnsToUse"]

# Columna con identificador unívoco del dato
idData = conf["columns"]["idColumn"]

# Separamos variables de encodings (supervisadas) del resto
lista_variables_encodings = conf["columns"]["encodings"]
resto_variables = conf["columns"]["rest"]


# META: Mejor que resto sean el resto, por si hay muchas. También, si hay muchas, sustituir
# luego desplegable por autocompletar (o configuración)

# Cuáles son categóricas y necesitan generar columnas numéricas asociadas?
# Si ya vienen en el csv como <nombre>#, no indicar aquí.
list_cl_str = conf["columns"]["categorical"]

# Títulos para hover, app y figura
appTitle = conf["appearance"]["appTitle"]
hoverTitle = conf["appearance"]["hoverTitle"]
figTitle = conf["appearance"]["figTitle"]

# Variables en hover
listaHover = conf["columns"]["hover"]

# Datos con posiciones precalculadas (ej: tSNE)
precalculatedPositionsFiles = conf["precalcEncodings"]["files"]
precalculatedPositionsNames = conf["precalcEncodings"]["names"]

##########################################################

# Donde se guardan los datos para descargarlos al cliente
savepath = './static/export'

# Leemos los datos principales
df = pd.read_csv(datapath + filename, index_col=0)

# Si hay columna para ordenar, ordenamos
if(sortbyColumn != ''):
    df.sort_values(by=[sortbyColumn],ignore_index=True, inplace=True)

# Si no tenemos lista con las variables resto (porque sean muchas, por ejemplo), 
# tratar de obtenerla automáticamente.
if resto_variables == []:
    ll=df.columns.to_list()
    resto_variables = [i for i in ll if i not in lista_variables_encodings]
    resto_variables = [i for i in resto_variables if '#' not in i]
    if idData in resto_variables:
        resto_variables.remove(idData)


# Generamos variables numéricas para las categóricas
for i in list_cl_str:
    clave = set(df[i])
    diccionario = dict(np.column_stack(
        (list(sorted(clave)), np.array(range(len(clave))).astype(object))))
    df[i+'#'] = [diccionario[j] for j in df[i]]

ll=df.columns.to_list()
list_cl_strnum = [i for i in ll if '#' in i]

# Añadimos los diferentes encodings
E = []
encoding_name = []

# Añadimos codificaciones pre-calculadas
i=0
for file in precalculatedPositionsFiles:
    pos =np.load(datapath + file)
    # Normalizamos, por si acaso no vienen...
    pos = pos/np.max(np.abs(pos))
    E.append(pos)
    encoding_name.append(precalculatedPositionsNames[i])
    i = i+1

# Generamos ahora encodings para las variables supervisadas. Circular para las categóricas, lineal para
# las numéricas (horizontal)
for i in lista_variables_encodings:
    print('generando codificaciones espaciales para variable ' + i + '...')

    if (i+'#' in list_cl_strnum):
        pos = df[[i+'#']].merge(circularEnc(df[i+'#'].unique(), i+'#'),
                            on=i + '#', how='left')[['posx', 'posy']].values
    else:
        pos = linearEnc(df, i, 'hor')
  
    encoding_name.append(i)
    E.append(pos)

# Ahora las cajas que permiten añadir encodings para el resto de variables

print('Creando cajas...')    
encoding_name.append(vbleEncodingHor)
E.append(linearEnc(df, vbleEncodingHor,'hor'))

encoding_name.append(vbleEncodingVer)
E.append(linearEnc(df, vbleEncodingVer, 'ver'))

if (columnas == []):
    columnas = df.columns.tolist()


print('Creando source...')    
# Preparamos el source
aux = pd.concat([
    df[columnas]],
      axis=1)

# fuente de datos
source = ColumnDataSource(aux)

# Utilizamos la vista, para reudcir el número de datos procesados
from bokeh.models import CDSView, BooleanFilter
if(__version__ < '3.0'):
    myView = CDSView(source = source, filters=[])
else:
    myView = CDSView()
    #myView.filter = BooleanFilter(booleans)

print('Inicializando cosas...')    
# Inicializaciones necesarias
selGA = []
selGB = []

################
# colores
###############
source.data['color'], TheColorMapper= getColorMap(df, colorbyColumn)

##############
# añadimos encodings
##############

print('Añadiendo encodings al source...')    

# añadimos los encodings al source
# los tenemos que apilar en forma de matriz (n_samples,2*num_encodings)
source.data['E'] = np.concatenate(E, axis=1).tolist()


def ResetEncodings(which=0, quantity=0.5):
    # nota: asignamos al encoding base E[which] un peso de quantity
    z = np.zeros(len(encoding_name))
    z[which] = quantity
    a = softmax(10*z)

    Epos = np.zeros([len(E[0]), 2])
    for i in range(len(encoding_name)):
        print('Encoding: '+ encoding_name[i] +'...')
        Epos = Epos + a[i]*E[i]

    source.data['pos_x'] = Epos[:, 0]
    source.data['pos_y'] = Epos[:, 1]

ResetEncodings(0,0.5) # Resetea los encodings, poniendo el 0 al 50%

##########################################


#######################################################
# FIGURAS BOKEH
#######################################################

##########################################
# Caja de coloreado
##########################################


var_color = CheckboxGroup(
    labels=["usar mapa de color:"], active=[])


def update_coloring(attrname, old, new):
    global plot
    from matplotlib import cm

    if(len(new) == 0):
        variable = colorbyColumn
        plot.legend.visible=True 
        bar.visible = False
    else:
        variable = select_coloreado.value
        plot.legend.visible=False 
        bar.visible = True

    
    colores, TheColorMapper = getColorMap(df,variable)
    bar.color_mapper = TheColorMapper
              
    # ticker = FixedTicker(ticks=[TheColorMapper.low, TheColorMapper.high])
    # formatter = CustomJSTickFormatter(code="""
    #     var data = { %d: '%s',  %d: '%s' }
    #     return data[tick]
    # """ %(TheColorMapper.low, str(TheColorMapper.low), TheColorMapper.high, str(TheColorMapper.high)) )
    # bar.formatter = formatter
    # bar.ticker = ticker


    source.data['color'] = colores
    return

var_color.on_change('active', update_coloring)

##########################################
# Caja para el parámetro de coloreado
##########################################

def update_color_var(attrname, old, new):
    var_color.active = [0]
    update_coloring(None, None, [0])
    

select_coloreado = Select(value=vbleEncodingHor, options=resto_variables, title="color:", width=180)
select_coloreado.on_change('value', update_color_var)


##########################################
# Actualiza el contenido del tooltip
# para el hover
#########################################
def actualiza_hover():
    str = '<style>.bk-tooltip>div:not(:first-child){display:none;}</style><b>%s: </b> @{%s} <br>'%(hoverTitle,hoverTitle)
    #lista_hover = lista_variables_encodings[1:] + \
    #    [resto_variables[0]] + resto_variables[2:11] + [resto_variables[-1]]

    listaHoverAux = listaHover.copy()
    if (vbleEncodingHor not in listaHoverAux):
        listaHoverAux.append(vbleEncodingHor)
    if (vbleEncodingVer not in listaHoverAux):
        listaHoverAux.append(vbleEncodingVer)
    for i in listaHoverAux:
        str = str + '<b>{}:</b> @{}<br>'.format(i, i)
    custom_hover.tooltips = str


###########################################
# FIGURA PRINCIPAL: PLOT DE LAS MUESTRAS
###########################################

if(__version__ >= '3.1.0'):
    plot = figure(name='mainplot', height=800, width=1000, title=figTitle,
              tools="crosshair,pan,reset,save,wheel_zoom,box_select,lasso_select",
              toolbar_location="above",
              output_backend='webgl',match_aspect=True)
else:
    plot = figure(name='mainplot', plot_height=800, plot_width=1000, title=figTitle,
              tools="crosshair,pan,reset,save,wheel_zoom,box_select,lasso_select",
              toolbar_location="above",
              output_backend='webgl',match_aspect=True)

#plt = plot.circle(x='pos_x', y='pos_y', color='color',
#                 source=source, view = myView, 
#                 size=6, #radius=0.5,
#                 alpha=0.7, nonselection_alpha=0.1, legend_group=colorbyColumn, 
#                 line_color='black', line_width=0.5)

plt = plot.scatter('pos_x', 'pos_y', color='color',
           source=source, view=myView, 
           # legend_group=colorbyColumn,
           size=6, alpha=0.7, 
           selection_color='red',
           nonselection_alpha=0.1, 
           line_color='#000000', 
           # line_width=0.5
           line_width=0.0
           )


# añadir colorbar a la figura
# ticker = FixedTicker(ticks=[TheColorMapper.low, TheColorMapper.high])
# formatter = CustomJSTickFormatter(code="""
#     var data = { %d: '%s',  %d: '%s' }
#     return data[tick]
# """%(TheColorMapper.low, str(TheColorMapper.low), TheColorMapper.high, str(TheColorMapper.high)) )
# bar = ColorBar(color_mapper = TheColorMapper, formatter = formatter, ticker = ticker, location=(0,0))

bar = ColorBar(color_mapper = TheColorMapper, location=(0,0))
plot.add_layout(bar, "left")

# META: Un poco chapuza para quitar la barra si es categórico...
if len(df[colorbyColumn].unique())<=20:
     bar.visible = False


# Vamos a crear una herramienta hover customizada para que se comporte como queremos
# mostrando los valores de las variables de encodings.
custom_hover = HoverTool()
actualiza_hover()
plot.tools.append(custom_hover)

# Prueba para ajustar el source al rango:

def event_callback(event):
    global myView

    # Algo pasa que usa el source en el servidor, no el del cliente
    # y, por tanto, los pos_x y pos_y del primer encoding, sin 
    # actualizar

    # Nos quedamos solo con los puntos visibles. Para eso
    # modificamo el filtro de la vista
 
    booleans = [True if ((x >= event.x0) and (x <= event.x1) and (y>=event.y0) and (y<=event.y1)) else False 
                for [x,y] in zip(source.data['pos_x'], source.data['pos_y'])]
    print(f"De {len(booleans)} Nos quedamos con {sum(booleans)}")
      
"""     if(__version__ < '3.0'):
        myView.filters = [BooleanFilter(booleans)]
    else:
        myView.filter = BooleanFilter(booleans) """


from bokeh.events import *
#plot.on_event(RangesUpdate, event_callback)

##############################################
# SPINNER
# Todo esto es para implementar el spinner, la
# rueda que gira cuando tenemos que realizar un
# cálculo que lleva tiempo.
##############################################

# Definimos el div con el gráfico animado
spinner_text = """
<!-- https://www.w3schools.com/howto/howto_css_loader.asp -->
<div class="loader">
<style scoped>
.loader {
    border: 16px solid #f3f3f3; /* Light grey */
    border-top: 16px solid #3498db; /* Blue */
    border-radius: 50%;
    width: 120px;
    height: 120px;
    animation: spin 2s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
</div>
"""
div_spinner = Div(text="", width=300, height=120)

# Y ahora dos funciones para mostrarlo y ocultarlo


def show_spinner():
    div_spinner.text = spinner_text


def hide_spinner():
    div_spinner.text = ""


def SelectionCallback(attrname, old, new):
    global source
    ss = df.iloc[new]
    if len(ss) > 0:
        # Informamos del número de elementos copiados
        tt = 'Seleccionados %d registros de %d datos diferentes.' % (
            len(ss), len(ss[hoverTitle].unique()))
    else:
        tt = ''
    div_spinner.text = tt


# Esto ha cambiado en la versión 1.3.4!!!!
# así que chequeamos la versión y usamos la llamada
# adecuada para instalar la callback
if(__version__ >= '1.3.4'):
    plt.data_source.selected.on_change('indices', SelectionCallback)
else:
    plt.data_source.on_change('selected', SelectionCallback)

#######################################################
# SLIDERS
#######################################################

sliders = []
for i in encoding_name:
    sliders.append(Slider(title=i, value=0, start=0,
                          end=1, step=0.01, width=200, height=39))

sliders[0].value = 0.5

# Callback para los sliders que realiza el morphing mediate la función Softmax.
# Implementación en javascript para que se ejecute localmente.
# De este modo es mucho más eficiente y la visualización es más
# suave.
# Recibe la fuente de datos, el mapa de encodings y una lista con los
# widgets slider para tratar.
CodigoJS = """
  // pesos para los encodings definidos en la interfaz
  var z =  s.map(x=>x.value)

  // aplicamos coeficiente de sensibilidad de morphing (sens = 10)
  z = z.map(x => x*10)  

  // calculamos función softmax para obtener los pesos "a"
  var ez      = z.map(x => Math.exp(x))       // exponentes de z_i
  var suma_ez = ez.reduce( (x,v) => x + v)    // suma de los exponentes de z_i 
  var a       = ez.map( x => x/suma_ez)       // exponentes de z_i / suma de exponentes de z_i

  // No sé por qué razón así parece ir mucho más rápido. 
  a = a.map(x => Math.round((x + Number.EPSILON) * 1000) / 1000)
   
  
  // calculamos encodings x pesos
  var Epos = []
  var data = source.data
  var N = data['E'].length
  for (var j=0; j< N; j++ )
  {
  Epos[j] = [0,0]
  for (var i=0; i< a.length; i++)
    {
    Epos[j][0] += a[i]*data['E'][j][2*i]
    Epos[j][1] += a[i]*data['E'][j][2*i+1]
    }
  }


  // actualizadmos coordenadas x,y en source
  data['pos_x'] = Epos.map(x => x[0])
  data['pos_y'] = Epos.map(x => x[1])
  
  source.data = data
  source.change.emit()

"""

# Instalamos la callback anterior
update_data = CustomJS(args=dict(source=source,  s=sliders), code=CodigoJS)
for s in sliders:
    s.js_on_change('value', update_data)


################################################################
# Cajas autocompletables o desplegables para elegir variable
################################################################

if autocomplete:
    select1 = AutocompleteInput(
        value=vbleEncodingHor, completions=resto_variables, title="hor:", width=180)
    select2 = AutocompleteInput(
        value=vbleEncodingVer, completions=resto_variables, title="ver:", width=180)
else:
    select1 = Select(
        value=vbleEncodingHor, options=resto_variables, title="hor:", width=180)
    select2 = Select(
        value=vbleEncodingVer, options=resto_variables, title="ver:", width=180)


# Función que actualiza los encodings cuando se modifican las variables
# que tienen encodings lineales horizontal y vertical.
# Esto toma tiempo, de forma que se hace en python y en dos partes.
# El proceso es un poco complejo, para que se muestren los spinners.
# Se realiza una callback normal para el control que:
#   1- muestra el spinner
#   2- instala una callback para el siguiente tick de reloj
#     que realiza el trabajo
#   3- esa callback realiza el trabajo y esconde el spinner
# Además se hacen otras cosas, como calcular los nuevos encodings,
# o deshabilitar los sliders para evitar interacciones del usuario mientras
# se realiza el trabajo.

# Función que actualiza los encodings en la fuente de datos en python.


def actualiza_encodings():
    import sys
    z = np.array([s.value for s in sliders])
    a = softmax(10*z)
    a = list(map(lambda x: round(x,3), a))

    Epos = np.zeros([len(E[0]), 2])
    for i in range(len(encoding_name)):
        Epos = Epos + a[i]*E[i]

    p1 = [i for i in zip(range(Epos.shape[0]), Epos[:, 0])]
    p2 = [i for i in zip(range(Epos.shape[0]), Epos[:, 1])]

    source.patch(dict(pos_x=p1, pos_y=p2))

    return


# Función auxiliar que deshabilita los sliders y las cajas de texto


def disable_sliders(b):
    for s in sliders:
        s.disabled = b
    select1.disabled = b
    select2.disabled = b

# Función que realiza el trabajo para el paso 3


def actualiza_worker():
    source.data['E'] = np.concatenate(E, axis=1).tolist()
    actualiza_encodings()
    actualiza_hover()
    disable_sliders(False)
    hide_spinner()


# Callbacks de los controles


def update_select1(attrname, old, new):
    encoding_name[-2] = new
    E[-2] = (linearEnc(df, new, 'hor'))
    sliders[-2].title = new
    global vbleEncodingHor
    vbleEncodingHor = new
    disable_sliders(True)
    show_spinner()
    curdoc().add_next_tick_callback(actualiza_worker)


def update_select2(attrname, old, new):
    encoding_name[-1] = new
    E[-1] = (linearEnc(df, new, 'ver'))
    sliders[-1].title = new
    global vbleEncodingVer
    vbleEncodingVer = new
    disable_sliders(True)
    show_spinner()
    curdoc().add_next_tick_callback(actualiza_worker)


# Asignamos las callbacks
select1.on_change('value', update_select1)
select2.on_change('value', update_select2)



##########################################
# Botones para tipo de encoding de los
# sliders
##########################################

tipo_encoding = []
for i in encoding_name:
    if (i + '#' in list_cl_strnum):
        sactive = 0
    else:
        sactive = 1
        
    tipo_encoding.append(RadioButtonGroup(
        labels=["circ", "hor", "ver"], active=sactive, orientation="horizontal", width=150, height=39)
    )

# Deshabilitar los que son de tSNE (no tiene sentido otra que circular)
for i in range(len(precalculatedPositionsNames)):
    tipo_encoding[i].disabled = True

# Poner la correspondiente para las dos últimas (cajas de variables no supervisadas)
# y deshabilitar
tipo_encoding[-1].active = 2
tipo_encoding[-1].disabled = True
tipo_encoding[-2].active = 1
tipo_encoding[-2].disabled = True

# Ahora la callback para todos

# Una función auxiliar para recalcular los encodings...
def recalcula_encodings(E):
    k = len(precalculatedPositionsNames)  # Tras los tSNE
    for i in lista_variables_encodings:
        print('recalculando encoding para ' + i + '...')
        tipo = tipo_encoding[k].active
        if (i + '#' in list_cl_strnum):
            name = i + '#'
        else:
            name = i

        if (tipo == 0):
            #pos = df[[name]].merge(circularEnc(max(df[name])+1, name),
            #                                    on=name, how='left')[['posx', 'posy']].values
            pos = df[[name]].merge(circularEnc(df[name].unique(), name),
                                                on=name, how='left')[['posx', 'posy']].values
            print('Circular')
        elif (tipo == 1):
            pos = linearEnc(df, name, 'hor')
            print('Horizontal')
        else:
            pos = linearEnc(df, name, 'ver')
            print('Vertical')

        E[k] = pos

        k = k+1
    return


def update_tipo_encoding(attrname, old, new):
    global source
    global E
    global df
    disable_sliders(True)
    show_spinner()
    recalcula_encodings(E)
    curdoc().add_next_tick_callback(actualiza_worker)
    return


for i in tipo_encoding:
    i.on_change('active', update_tipo_encoding)




#######################################################
# DESCARGA DE DATOS
#######################################################

############################################
# Enviar resultados guardados al cliente
#############################################

# Esto necesita una callback en javascript
JScode_fetch = """
debugger
var filename = t.text;
console.log('Entering saving code..')
if (filename != '') {
    console.log('Saving..'+filename)
   
    try {
            fetch('/'+'%s'+filename, {
                cache: 'no-store',
                method: 'GET',
                headers: {
                    // 'Authorization': 'Bearer ' + sessionStorage.getItem('token')
                },
            }).then(
                data => {
                    return data.blob();
                }
            ).then(
                response => {
                    console.log(response.type);
                    const dataType = response.type;
                    const binaryData = [];
                    binaryData.push(response);
                    const downloadLink = document.createElement('a');
                    downloadLink.href = window.URL.createObjectURL(new Blob(binaryData, { type: dataType }));
                    downloadLink.setAttribute('download', filename);
                    document.body.appendChild(downloadLink);
                    downloadLink.click();
                    downloadLink.remove();
                }
            )

        } catch (e) {
            addToast('Error inesperado.', {appearance: 'error', autoDismiss: true});
    }
}
t.text=''
console.log('Exiting saving code..')
""" %(savepath)

# y un control dummy para asociar esa función
div_dummy = Div(text="", width=300, height=120, visible=False)
div_dummy.js_on_change('text', CustomJS(args=dict(t=div_dummy),
                                                  code=JScode_fetch))



##########################################
# Botón para calcular influencias
# de variables en separación de grupos A y B
#########################################

# Función que calcula las influencias de las variables (pasados en lista_diff)
# en el agrupamiento tSNE utilizando la regresión logística
# Esto toma tiempo, de forma que se hace en dos partes.
# El proceso es un poco complejo, para que se muestren los spinners.
# Se realiza una callback normal para el control que:
#   1- comprueba que ambos grupos tengan datos muestra el spinner
#   2- instala una callback para el siguiente tick de reloj
#     que realiza el trabajo
#   3- esa callback realiza el trabajo y esconde el spinner


def influencias_worker():
    global selGA
    global selGB
    global lista_diff
    undf = CalculaVectorOrdenado(selGA, selGB, lista_diff)

    # Vamos a imprimir las primeras variables, según la ordenación anterior
    l = undf.index
    tabla = '''<style type="text/css">
  .tg  {border-collapse:collapse;border-spacing:0;}
  .tg td{font-family:Arial, sans-serif;font-size:11px;padding:0px 0px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
  .tg th{font-family:Arial, sans-serif;font-size:11px;font-weight:normal;padding:0px 0px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
  .tg .tg-fila{border-color:#ffffff;text-align:center;vertical-align:top}
  .tg .tg-tit{font-weight:bold;border-color:#ffffff;text-align:center;vertical-align:top}
  </style>
  '''
    tabla += '<table class="tg"><tr><th class="tg-tit">Var</th><th class="tg-tit">A (μ ± σ)</th><th class="tg-tit">B (μ ± σ)</th><th class="tg-tit">Relevance</th></tr>'
    #k = 0
    #while (undf.iloc[k, 0] >= 0.1 and k < 100):
    #    tabla += '<tr><td class="tg-fila">%s</td><td class="tg-fila">%2.3g ± %2.3g</td><td class="tg-fila">%2.3g ± %2.3g</td><td class="tg-fila">%2.3g</td class="tg-fila"></tr>' % (l[k],
    #                                                                                                                                                                                 df.iloc[selGA, :][l[k]].mean(
    #                                                                                                                                                                                     axis=0), df.iloc[selGA, :][l[k]].std(axis=0),
    #                                                                                                                                                                                 df.iloc[selGB, :][l[k]].mean(
    #                                                                                                                                                                                     axis=0), df.iloc[selGB, :][l[k]].std(axis=0),
    #                                                                                                                                                                                 undf.iloc[k, 0])
    #    k = k+1

    k = 0
    while k < len(l) and k < 100:
        tabla += '<tr><td class="tg-fila">%s</td><td class="tg-fila">%2.3g ± %2.2g</td><td class="tg-fila">%2.3g ± %2.2g</td><td class="tg-fila">%2.2g</td class="tg-fila"></tr>' % (l[k],
                                                                                                                                                                                     df.iloc[selGA, :][l[k]].mean(
                                                                                                                                                                                         axis=0), df.iloc[selGA, :][l[k]].std(axis=0),
                                                                                                                                                                                     df.iloc[selGB, :][l[k]].mean(
                                                                                                                                                                                         axis=0), df.iloc[selGB, :][l[k]].std(axis=0),
                                                                                                                                                                                     undf.iloc[k, 0])
        k = k+1                                                                                                                                                                                          
                         

    tabla += '</table>'
    notifications.text = tabla
    disable_sliders(False)
    hide_spinner()


def InfluenciasCallback(event):
    global selGA
    global selGB
    global lista_diff
    
    lista_diff = resto_variables

    if (len(selGA) == 0):
        notifications.text = 'Group A is empty'
        return

    if (len(selGB) == 0):
        notifications.text = 'Group B is empty'
        return
    show_spinner()
    disable_sliders(True)
    curdoc().add_next_tick_callback(influencias_worker)
    return


boton_influencias = Button(
    label="Diff A  ► B ", button_type="success", sizing_mode = "scale_width")
boton_influencias.on_click(InfluenciasCallback)


#############################################################
# OTROS CONTROLES
############################################################

#########################################
# Notificaciones
#########################################

if(__version__ >= '3.1.0'):
    notifications = Div(text='', styles={'title': 'Salida.', 'border': 'double', 'overflow-y': 'scroll',
                                    'width': '300px', 'height': '300px'})
else:
    notifications = Div(text='', style={'title': 'Salida.', 'border': 'double', 'overflow-y': 'scroll',
                                    'width': '300px', 'height': '300px'})


##########################################
# Botón para asignar la selección
# actual al grupo A
#########################################

def SeleccionGACallback(event):
    global selGA
    selGA = source.selected.indices
    if len(selGA) > 0:
        notifications.text = '%d registers selected to group A' % (
            len(selGA))
        source.selected.indices = []
        boton_gA.label = 'Selection to group A(%d)' % (len(selGA))
    else:
        boton_gA.label = 'Selection to group A'


boton_gA = Button(label="Selection to group A", button_type="success", sizing_mode = "scale_width")
boton_gA.on_click(SeleccionGACallback)

##########################################
# Botón para asignar la selección
# actual al grupo B
#########################################


def SeleccionGBCallback(event):
    global selGB
    selGB = source.selected.indices
    if len(selGB) > 0:
        notifications.text = '%d registers selected to group B' % (
            len(selGB))
        source.selected.indices = []
        boton_gB.label = 'Selection to group B (%d)' % (len(selGB))
    else:
        boton_gB.label = 'Selection to group B'


boton_gB = Button(label="Selection to group B", button_type="success", sizing_mode = "scale_width")
boton_gB.on_click(SeleccionGBCallback)



##########################################
# Botón para guardar la selección
# actual en un archivo excel
#########################################

def SaveExcel_worker():
    filename = "selected_data.xlsx"
    df.iloc[source.selected.indices].T.to_excel(savepath+filename)
    div_dummy.text = filename
    disable_sliders(False)
    hide_spinner()


def SeleccionSaveExcelCallback(event):
    if(len(source.selected.indices) == 0):
        notifications.text = 'Error: Emtpy selection.'
    else:
        show_spinner()
        disable_sliders(True)
        notifications.text = 'Saving %d registers to %s. This can take time (approx. 30 sec / 40 registers).' % (
            len(source.selected.indices), "selected_data.xlsx")
        curdoc().add_next_tick_callback(SaveExcel_worker)
    return


boton_save = Button(label="Save Selection to Excel", button_type="success", sizing_mode = "scale_width")
boton_save.on_click(SeleccionSaveExcelCallback)


##########################################
# Botón para guardar la selección
# actual como un dataframe en hdf
# útil para reproducir figuras
#########################################

def SaveHdf_worker():
    filename = "selected_data.hdf"
    
    actualiza_encodings()
    l=listaHover.copy()
    l.append('pos_x')
    l.append('pos_y')
    print(l)
    tempdf = pd.DataFrame(source.data)[l]
    
    tempdf.iloc[source.selected.indices].to_hdf(savepath+filename, key='raiz')
    div_dummy.text = filename

    disable_sliders(False)
    hide_spinner()


def SeleccionSaveHdfCallback(event):
    if(len(source.selected.indices) == 0):
        notifications.text = 'Error: Empty selection.'
    else:
        show_spinner()
        disable_sliders(True)
        notifications.text = 'Saving %d registers to %s.' % (
            len(source.selected.indices), 'selected_data.hdf')
        curdoc().add_next_tick_callback(SaveHdf_worker)
    return


boton_savehdf = Button(label="Save Selection to Hdf", button_type="success", sizing_mode = "scale_width")
boton_savehdf.on_click(SeleccionSaveHdfCallback)

# añadido Nacho y Chema 2024-07-17
boton_savehdf.disabled = True


#######################################################
# CUADRO DE TEXTO PARA INTERPRETAR COMANDOS
#######################################################
textinput = TextInput(title='Command (HELP)')
textinput.value = ''

# Vamos con algunas funciones auxiliares involucradas...

# Función que calcula un test anova (one-way anova) para un
# gen o mirna en los dos conjuntos de muestras A y B
def anova(var1):
    import scipy.stats as stats

    # Algunas comprobaciones, por si acaso...
    if(len(selGA) == 0 or len(selGB) == 0):
        tt = 'Error: Groups A and B cannot be empty.'
    elif ((var1 in df.iloc[selGA]) == False):
        tt = 'Error: variable not found.'
    else:
        # Calculamos el test y preparamos la cadena con el resultado
        r = stats.f_oneway(df.iloc[selGA][var1], df.iloc[selGB][var1])
        if(r.pvalue < 0.001):
            tt = '<b>ANOVA (%s):</b><br>N = %d<br>F-value = %3.2f<br>p-value < 0.001' % (
                var1, len(selGA) + len(selGB), r.statistic)
        else:
            tt = '<b>ANOVA (%s):</b><br>N = %d<br>F-value = %3.2f<br>p-value = %3.3f' % (
                var1, len(selGA) + len(selGB), r.statistic, r.pvalue)
    # Mostramos el resultado
    div_spinner.text = tt

# Función que calcula la correlación de Pearson para dos genes o miRNA
# en un conjunto de muestras: A, B o la selección actual (por defecto)
# Recibe el nombre de dos genes y las cadenas 'A', 'B' o 'SEL'


def correlacion(var1, var2, grupo='SEL'):
    import scipy.stats as stats

    # Preparamos el grupo de índices
    igrupo = []
    if(grupo.upper() == 'A'):
        igrupo = selGA
    elif(grupo.upper() == 'B'):
        igrupo = selGB
    elif(grupo.upper() == 'SEL'):
        igrupo = source.selected.indices

    if (not((var1 in df) and (var2 in df))):
        tt = 'Error: variable not found.'
    else:
        print('Correlation %s, N= %d' % (grupo, len(igrupo)))
        print('Vars: %s y %s' % (var1, var2))

        # Si el grupo elegido no esta vacío, se hacen los cálculos...
        if(len(igrupo) == 0):
            tt = 'Error: Empty group.'
        else:
            r = stats.pearsonr(df.iloc[igrupo][var1], df.iloc[igrupo][var2])
            if(r[1] < 0.001):
                tt = '<b>Corr (%s<-->%s):</b><br>r = %3.2f<br>p-value < 0.001' % (
                    var1, var2, r[0])
            else:
                tt = '<b>Corr (%s<-->%s):</b><br>r = %3.2f<br>p-value = %3.3f' % (
                    var1, var2, r[0], r[1])

    # Imprimimos resultados
    div_spinner.text = tt

# Selecciona muestras que coincidan con la lista de patrones.
# ej: SEL= TCGA-A8, -01


def selecciona(lista):
    todos = []
    for element in lista:
        todos = todos+[i for i in df[idData] if element.strip() in i]

    idx = [np.where(df[idData] == i)[0][0] for i in todos]

    # Ojo que idx aquí es de tipo numpy.int64, y luego falla al aplicar patch
    # lo convertimos a int...
    source.selected.indices = [int(i) for i in idx]
    source.data = dict(source.data)


#########################################
# Para las gráficas extra
#########################################

if(__version__ >= '3.1.0'):
    extra_graphs = figure(name='extra', height=300, width=300, 
              tools="save",
              toolbar_location="above",
              output_backend='webgl')
else:
    extra_graphs = figure(name='extra', plot_height=300, plot_width=300, 
              tools="save",
              toolbar_location="above",
              output_backend='webgl')

def make_ploth(df_plot, color='navy', keep = False):
    ind=0
    #colors=['navy','red','green']
    
    if extra_graphs.renderers != [] and not keep:
        extra_graphs.renderers = []
    hist, edges = np.histogram(df_plot, density=True, bins="rice")
    extra_graphs.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
    fill_color=color, line_color="white", alpha=0.5)
 
    if not keep:
        extra_graphs.y_range.start = 0  
        extra_graphs.xaxis.axis_label = 'x'
        extra_graphs.yaxis.axis_label = 'Pr(x)'
        extra_graphs.grid.grid_line_color="white"
    
    return


def histograma(var):
    if (not (var in df)):
        div_spinner.text = 'Error: variable not found.'
    else:
        if(len(selGA) != 0):
            print('Histograma de %s sobre grupo %s, N= %d' % (var, 'A', len(selGA)))
            make_ploth(df.iloc[selGA][var], color='blue')
        if(len(selGB) != 0):
            print('Histograma de %s sobre grupo %s, N= %d' % (var, 'B', len(selGB)))
            make_ploth(df.iloc[selGB][var], color='red', keep = True)
        if(len(source.selected.indices) != 0):
            print('Histograma de %s sobre grupo %s, N= %d' % (var, 'SEL', len(source.selected.indices)))
            make_ploth(df.iloc[source.selected.indices][var], color='green', keep = True)
    return


def filtra(var):
    # Nos quedamos solo con los puntos visibles. Para eso
    # modificamos el filtro de la vista

    div_spinner.text = 'Filtering data.'
    if var=="":
        filter_to_use= ~IndexFilter(indices=[])
    elif(var=="SEL"):
        filter_to_use=IndexFilter(indices=source.selected.indices)
       
        
    if(__version__ < '3.0'):
        myView.filters = [filter_to_use]
    else:
        myView.filter = filter_to_use

    return

# Función auxiliar que modifica una expresión para que las apariciones de variables (pasadas como una lista)
# se sustituyan por accesos al dataframe (df['variable])
import re

def sustituir_expresion(expresion, variables):
    # Crear un patrón de búsqueda para encontrar las variables en la expresión
    patron = re.compile(r'\b(' + '|'.join(variables) + r')\b')
    
    # Sustituir las variables encontradas por su forma FE correspondiente
    expresion_sustituida = patron.sub(lambda x: f'df[\'{x.group(1)}\']', expresion)
    
    return expresion_sustituida


# Colorear utilizando el valor resultante de una expresión que involucre variables
def colorea(expression):
    global TheColorMapper

    # Primero hay que modificar las variables para que se acceda como columnas del dataframe df
    vals = []
    try:
        res = sustituir_expresion(expression,df.columns.to_list())
        print(res)
        vals = eval(res)
    except:
        div_spinner.text = 'Error: incorrect expression.'
        return
    
    # Crearemos un colormap un poco robusto, por si hay outlayers...
    [min,max]=np.quantile(vals,[0.05,0.95])
    colors = [RGB(*cm.viridis(i, bytes=True)[:4]) for i in Normalize(vmin=min, vmax=max) (vals)]
    source.data['color']= colors
    TheColorMapper = LinearColorMapper(palette = 'Viridis256',
                                 low = min, 
                                 high = max)
    # colors = [RGB(*cm.turbo(i, bytes=True)[:4]) for i in Normalize(vmin=min, vmax=max) (vals)]
    # source.data['color']= colors
    # TheColorMapper = LinearColorMapper(palette = 'Turbo256',
    #                              low = min, 
    #                              high = max)
    bar.color_mapper = TheColorMapper
    plot.legend.visible=False 
    bar.visible = True
    #var_color.active = []

    div_spinner.text = f'Evaluated {res} - Done.'

# Muestra la ayuda en el cuadro de notificación
def ayuda():
    notifications.text = '''
<p><strong>ANOVA=&lt;var&gt;</strong></p>
<p>ANOVA test for the given variable in groups A and B.</p>
<p><strong>CORR=&lt;var1&gt;,&lt;var2&gt;[,A|B|SEL]</strong></p>
<p>Pearson correlation coefficient and p-value for the given vars in groups A, B, or selected registers (SEL=default).</p>
<p><strong>SEL=&lt;patron&gt;[,&lt;patron&gt;]*</strong></p>
<p>Select registers by partial ID.</p>
<p><strong>FILTER=[&lt;SEL&gt;]</strong></p>
<p>Filters the data to the current selection (if SEL is passed) or removes filter if no argument is passed.</p>
<p><strong>HIST=&lt;var&gt;</strong></p>
<p>Draws the histogram for the selected variable on those non-empty groups (A, B and SEL)</p>
<p><strong>COLOR=&lt;expr&gt;</strong></p>
<p>Sets the color as the result of the math expression (involving variables)</p>
<p><strong>RESET</strong></p>
<p>Resets encodings</p>
'''
    return

# Callback que interpreta el comando entrado por el usuario.
# De momento:
# HELP
# ANOVA=<var>
# CORR=<var1>,<var2>[,<conjunto = 'A'|'B'|'SEL'>]
# SEL=<patron>[,<patron>]
# HIST=<var>
# FILTER=<conjunto = 'A'|'B'|'SEL'>
# COLOR=<expr = f({<var>})>
# RESET resetea encodings

def interpreta_comando(attrname, old, new):
    if(new == ''):
        return

    div_spinner.text = ''
    
    st = textinput.value.split('=')
    comando = st[0].strip().upper()

    if (comando == 'ANOVA'):
        anova(st[1].strip())
    elif (comando == 'CORR'):
        s = st[1].split(',')
        if(len(s) == 3):
            correlacion(s[0].strip(), s[1].strip(), grupo=s[2].strip())
        elif(len(s) == 2):
            correlacion(s[0].strip(), s[1].strip())
        else:
            div_spinner.text = 'Syntax Error.'
    elif (comando == 'SEL'):
        selecciona(st[1].split(','))
    elif (comando == 'HIST'):
        histograma(st[1].strip())
    elif (comando == 'HELP'):
        ayuda()
    elif (comando == 'FILTER'):
        filtra(st[1].strip())
    elif(comando == 'COLOR'):
        colorea(st[1].strip())
    elif(comando == 'RESET'):
        #ResetEncodings(0,0.5)
        for s in sliders:
            s.value = 0
        sliders[0].value = 0.5
    else:
        div_spinner.text = 'Error: unknown command.'


# Instalamos la callback
textinput.on_change('value', interpreta_comando)

##########################################
# Botón logout
#########################################
boton_logout = Button(label="logout", button_type="primary", sizing_mode = "scale_width")
boton_logout.js_on_event("button_click", CustomJS(code=f"window.location.href='{curdoc().session_context.request.path}{curdoc().session_context.logout_url}'"))

##########################################
# Layout completo
##########################################
# Controles para el morphing en dos columnas: sliders y selección del tipo de encoding.
# Añadimos al final las cajas con la selección del gen para encodings horizontal y vertical
morphing_controls = row(
    [column(sliders+[select1]), column(tipo_encoding+[select2])])

# Agrupamos en una columna los botones, la caja de notificaciones, la entrada de comandos y
# el div donde aparece el spinner (y algunos resultados)
#buttons_and_commands = column([boton_tsne, boton_save, boton_savehdf, boton_gA, boton_gB, boton_influencias,
                               #boton_influencias2, notifications, textinput, div_spinner])

buttons_and_commands = column([boton_logout, boton_save, boton_savehdf, boton_gA, boton_gB, boton_influencias, notifications, textinput, div_spinner])                               

# Controles que van debajo del plot. En principio la caja de coloreado por nivel de expresión
# de un gen. Como es un checkbox y una caja de texto, agrupamos ambos en una columna
#coloring = row(column([var_color, select_coloreado]), mark_types)
bottom = row(column([var_color, select_coloreado]))

# Por fin, el layout...
mylayout = layout([
    [plot, morphing_controls, buttons_and_commands, extra_graphs],
    [bottom, div_dummy]
])

curdoc().add_root(mylayout)
curdoc().title = appTitle
    
