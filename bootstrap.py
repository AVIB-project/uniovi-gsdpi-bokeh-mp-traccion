import os
import logging

from tornado.web import StaticFileHandler

from bokeh.server.server import Server
from bokeh.server.auth_provider import AuthModule
from bokeh.command.util import build_single_handler_application

# Bokeh Server arguments
app = "uniovi-gsdpi-bokeh-mp-traccion"
app_prefix = "MP"
app_port = 5006
app_title = "MP traccion"
app_logo = "logo_gsdpi.png"
app_background = "login_background.png"
cookie_secret = "gsdpi"
websocket_origin = ["avispe.edv.uniovi.es:80", "localhost:" + str(app_port)]
basic_username = "ingeteam"
basic_password = "M0rph1ng_"
login_level = logging.DEBUG

# Set Loggin level for bokeh server
logging.getLogger("bokeh").setLevel(login_level)

# Define Bokeh Authentication Module
auth_provider = AuthModule("./auth/auth.py")

# Define Bokeh application
bokeh_app = build_single_handler_application("./main.py")

# Define the server settings, including the cookie_secret
server_settings = {
    "cookie_secret": cookie_secret,
}

# Define the allowed WebSocket origins (adjust for your domains and ports)
websocket_origin = websocket_origin

# Define the static path
static_path = os.path.join(os.path.dirname(__file__), "static")

# Embed the Bokeh server in the Tornado application with Authenticacion
bokeh_server = Server(
    {"/": bokeh_app},
    port=app_port,
    prefix=app_prefix,
    auth_provider=auth_provider,
    allow_websocket_origin=websocket_origin,
    extra_patterns=[
        (r"/" + app + "/static/(.*)", StaticFileHandler, {"path": static_path})
    ],
    **server_settings)

# Configure Auth Provider arguments
auth_provider._module.login_url = f"/{app_prefix}/login"
auth_provider._module.logout_url = f"/{app_prefix}/logout"
auth_provider._module.app = app
auth_provider._module.app_title = app_title
auth_provider._module.prefix = app_prefix
auth_provider._module.style_path = "/" + app_prefix + "/" + app + "/static/css/styles.css"
auth_provider._module.logo_image = "/" + app_prefix + "/" + app + "/static/images/" + app_logo
auth_provider._module.background_image = "/" + app_prefix + "/" + app + "/static/images/" + app_background
auth_provider._module.basic_username = basic_username
auth_provider._module.basic_password = basic_password

# Start Bokeh Server
bokeh_server.start()
bokeh_server.io_loop.start()