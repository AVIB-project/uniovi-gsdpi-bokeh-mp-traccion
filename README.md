# Description
A PoC with bokeh server with a custom basic login view

## STEPS

- **STEP01**: clone the template auth bokeh repository as project name alias

```
$ git clone https://gsdpi@dev.azure.com/gsdpi/avib/_git/uniovi-gsdpi-bokeh-template uniovi-gsdpi-bokeh-mp-traccion
```

- **STEP02**: create virtual env for your project and activate it

```
$ python3 -m venv .venv
$ source .venv/bin/activate
```

- **STEP03**: Install default python modules in your project

```
(.venv)$ pip install -r requirements.txt
```

- **STEP04**: Install extra python dependencies for your application

```
(.venv)$ pip install matplotlib
(.venv)$ pip install scipy
(.venv)$ pip install scikit-learn
(.venv)$ pip install openpyxl
```

- **STEP05**: Freeze all modules used by your application

```
(.venv)$ pip freeze > requirements.txt
```

- **STEP06**: Configure Application

Edit bootstrap Bokeh Server and set these arguments:

```
app = "uniovi-gsdpi-bokeh-template"                                          --> App main folder bokeh application (must b ethe same as main folder)
app_prefix = "MP"                                                            --> Prefix for your bokeh application
app_port = 5006                                                              --> Port for your bokeh application
app_title = "MP traccion"                                                    --> Title for your bokeh application
app_logo = "logo_gsdpi.png"                                                  --> Logo for your bokeh application
app_background = "login_background.png"                                      --> Background for your bokeh application
cookie_secret = "gsdpi"                                                      --> Cookie secret for your bokeh application
websocket_origin = ["avispe.edv.uniovi.es:80", "localhost:" + str(app_port)] --> Web Origins Domains for your bokeh application
basic_username = "<USERNAME>"                                                --> Username credentials for your bokeh application
basic_password = "<PASSWORD>"                                                --> Password credentials for your bokeh application
login_level = logging.DEBUG                                                  --> Logging level or your bokeh application
```

- **STEP07**: Execute application and debug

```
$ python boostrap.py
```

- **STEP08**: Build the docker image

Exec this command to build:

```
$ docker build -t uniovi-gsdpi-bokeh-mp-traccion:1.0.0 .
```

- **STEP09**: run the docker container locally

Exec this command to run the container:

```
$ docker run --rm --name uniovi-gsdpi-bokeh-mp-traccion:1.0.0 -p 5006:5006 uniovi-gsdpi-bokeh-mp-traccion:1.0.0
```

- **STEP10**: tag image docker image to be uploaded to azure container registry

```
$ docker tag uniovi-gsdpi-bokeh-mp-traccion:1.0.0 avibdocker.azurecr.io/uniovi-gsdpi-bokeh-mp-traccion:1.0.0
```

- **STEP11**: push image docker image

```
$ docker push avibdocker.azurecr.io/uniovi-gsdpi-bokeh-mp-traccion:1.0.0
```