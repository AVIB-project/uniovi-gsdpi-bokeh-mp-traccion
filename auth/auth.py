import tornado
from tornado.web import RequestHandler

# Define Auth Module arguments configured from bootstrap
app_title = ""
prefix = ""
style_path = ""
logo_image = ""
background_image = ""
basic_username = ""
basic_password = ""

# could define get_user_async instead
def get_user(request_handler):
    return request_handler.get_signed_cookie("user")

# could also define get_login_url function (but must give up LoginHandler)
login_url = "/login"

# optional login page for login_url
class LoginHandler(RequestHandler):
    def get(self):
        try:
            error_message = self.get_argument("error")
        except Exception:
            error_message = ""

        self.render("login.html",                     
                    prefix=prefix,
                    app_title = app_title, 
                    style_path=style_path,
                    logo_image=logo_image,
                    background_image=background_image,
                    errormessage=error_message)

    def check_permission(self, username, password):
        # Basic authentication        
        if username == basic_username and password == basic_password:
            return True

        return False

    def post(self):
        username = self.get_argument("username", "")
        password = self.get_argument("password", "")

        auth = self.check_permission(username, password)

        if auth:
            self.set_current_user(username)
            self.redirect("/" + prefix)
        else:
            error_message = "?error=" + tornado.escape.url_escape("Login incorrect")
            self.redirect(login_url + error_message)

    def set_current_user(self, user):
        if user:
            self.set_signed_cookie("user", tornado.escape.json_encode(user))
        else:
            self.clear_cookie("user")

# optional logout_url, available as curdoc().session_context.logout_url
logout_url = "/logout"

# optional logout handler for logout_url
class LogoutHandler(RequestHandler):
    def get(self):
        self.clear_cookie("user")
        self.redirect("/" + prefix)
