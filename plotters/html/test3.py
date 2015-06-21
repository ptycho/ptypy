                                                ###############################
                                                #       Bennie Jaderberg      #
                                                #       28/07/2014            #
                                                ###############################
import threading
import webbrowser
from wsgiref.simple_server import make_server
import random

# Choose the html/javascript frontend we will be working with
FILE = 'frontend.html'
# Choose the local port to use
PORT = 8080

# Define a class to randomly select a stored image
def get_image_file():
    
    # Define an array of stored images
    svg_file_list = ['Desktop/new_server/x1.svg', 'Desktop/new_server/x2.svg', 'Desktop/new_server/x3.svg', 'Desktop/new_server/x4.svg']
    # Choose one at random
    svg_file = random.choice(svg_file_list)
    return svg_file


def test_app(environ, start_response):
    if environ['REQUEST_METHOD'] == 'POST':                 # If we receive a "POST" request from the webpage
        try:                                                # (i.e an interactive request, not from the URL)
            response_body = file(get_image_file()).read()   # set our response as the desired image
      
        except:
            response_body = "An error has occurred" 
            
        # Define the type of content being sent to the browser and return it
        status = '200 OK'
        headers = [('Content-type', 'image/svg+xml'), ('Content-Length', str(len(response_body)))]
        start_response(status, headers)
        return [response_body]    
    
    # If we receive any other type of request such as a "GET" request, re-open the html file.
    else:
        response_body = open(FILE).read()
    headers = [('Content-type', 'text/html'), ('Content-Length', str(len(response_body)))]
    status = '200 OK'
    start_response(status, headers)
    return [response_body]


def open_browser():
    # Start a browser after waiting for half a second.
    def _open_browser():
        webbrowser.open('http://localhost:%s/%s' % (PORT, FILE))
    thread = threading.Timer(0.5, _open_browser)
    thread.start()


def start_server():
    # Start the server
    httpd = make_server("", PORT, test_app)
    httpd.serve_forever()

if __name__ == "__main__":
    open_browser()
    start_server()
